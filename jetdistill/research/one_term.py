"""S14 — one term per input per value neuron (the user's goal): every input enters a neuron through a single term —
x, max(0, x − θ) or max(0, θ − x) at one of its 5 thresholds. Start from a per-head model (default S5rpc): for each
(head, neuron, input) the single term that best reproduces that input's current piece (least squares over jets, on the
pooled terms), then the kept coefficients re-tuned toward ParT's probabilities. The "below" hinges come from the pooled
terms already there: Σ α_i max(0, θ − x_i) = θ (1 − α_cls) − Σ α_i x_i + Σ α_i max(0, x_i − θ).
Saves {tag}1_model.npz: W over the extended basis [x (nv), x > θ_k (5 nv), θ_k > x (5 nv), α_cls, 1].

  python -m jetdistill.research.one_term [TAG epochs]"""
import json, sys, time
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import extract, downstream, phi_pooled, rows_of_split, OUT
from .heads_eval import uniformize


def extend(P, kn, nv):
    """(n, 16, 6 nv + 2) pooled terms → (n, 16, 11 nv + 2): + the pooled 'below' hinges"""
    lin, gt, acls, one = P[..., :nv], P[..., nv:6 * nv], P[..., 6 * nv:6 * nv + 1], P[..., 6 * nv + 1:]
    lt = np.concatenate([kn[k][None, None] * (1 - acls) - lin + gt[..., k * nv:(k + 1) * nv] for k in range(5)], -1)
    return np.concatenate([lin, gt, lt, acls, one], -1).astype(np.float32)


def cols_of(f, nv):
    return [f] + [nv + k * nv + f for k in range(5)] + [6 * nv + k * nv + f for k in range(5)]


def run(tag='S5rpc', epochs=150, n_fit=100000, n_dev=20000, lr=1e-4, lam=0.01, device='mps', log=print):
    import torch
    t0 = time.time(); Pz = dict(np.load(OUT / f'{tag}_model.npz')); W0, kn, uni = Pz['W'], Pz['kn'], str(Pz['uniform']); blocks = [int(c) - 1 for c in uni]
    model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    Of, Af, Mf, Lf, _ = extract(model, 'fit', n_fit, device); _, Ad, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Af, Ad = uniformize(Af, Mf, blocks), uniformize(Ad, Md, blocks); Jf, Jd = jets('full', 'fit'), jets('full', 'dev')
    Pf, _ = phi_pooled(Jf, rows_of_split('fit', n_fit), Af, kn); Pd, _ = phi_pooled(Jd, rows_of_split('dev', n_dev), Ad, kn)
    nv = (Pf.shape[2] - 2) // 6; knv = kn[:, :nv]; Ef, Ed = extend(Pf, knv, nv), extend(Pd, knv, nv); K = Ef.shape[2]
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)
    def agree(W):
        Wt = T(W) if not torch.is_tensor(W) else W
        with torch.no_grad():
            return float((torch.cat([downstream(model, *torch.einsum('nhk,hko->nho', T(Ed[a:a + 5000]), Wt).split(8, 1)).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
    Wst = np.zeros((16, K, 16), np.float32); Wst[:, :6 * nv] = W0[:, :6 * nv]; Wst[:, K - 2:] = W0[:, 6 * nv:]       # the start model in the extended basis
    a0 = agree(Wst); log(f'  {tag}: {100 * a0:.2f}% ({int((W0[:, :6 * nv] != 0).sum())} nonzero term coefficients)')
    # one term per (head, neuron, input): the candidate that best reproduces the input's piece over the jets
    W1 = np.zeros_like(Wst); W1[:, K - 2:] = Wst[:, K - 2:]; S = Ef[:20000]; choice = np.full((16, 16, nv), -1)
    for h in range(16):
        for o in range(16):
            for f in range(nv):
                c6 = [f] + [nv + k * nv + f for k in range(5)]; w = Wst[h, c6, o]
                if not np.any(w): continue
                y = S[:, h, c6] @ w; best = (np.inf, None, 0.0, 0.0)
                for j in cols_of(f, nv):
                    x = S[:, h, j]; xc = x - x.mean(); den = xc @ xc
                    if den < 1e-12: continue
                    a = float(xc @ (y - y.mean()) / den); err = float(((y - y.mean()) - a * xc) @ ((y - y.mean()) - a * xc))
                    if err < best[0]: best = (err, j, a, float(y.mean() - a * x.mean()))
                if best[1] is None: continue
                W1[h, best[1], o] = best[2]; W1[h, K - 1, o] += best[3]; choice[h, o, f] = best[1]
    a1 = agree(W1); nterm = int((W1[:, :K - 2] != 0).sum()); log(f'  one term per input (least squares): {100 * a1:.2f}% ({nterm} terms), {time.time() - t0:.0f} s')
    # re-tune the kept coefficients (and the class-token / bias columns), coefficients scaled by their column's spread
    mask = (W1 != 0).astype(np.float32); mask[:, K - 2:] = 1; sdc = Ef[:20000].std(0) + 1e-6                     # (16, K)
    V = torch.nn.Parameter(T(W1 * sdc[..., None])); Mk = T(mask); Sd = T(sdc)[..., None]; opt = torch.optim.Adam([V], lr)
    Eft = T(Ef); pf = torch.softmax(T(Lf), 1); Oft = T(Of.transpose(1, 0, 2, 3).reshape(n_fit, 16, 16)); vo = Oft.var(0) + 1e-6
    Wof = lambda: V * Mk / Sd; best = (a1, W1.copy(), 0); path = [(0, a1)]
    for ep in range(epochs):
        for a in np.random.default_rng(ep).permutation(np.arange(0, n_fit, 5000)):
            o = torch.einsum('nhk,hko->nho', Eft[a:a + 5000], Wof()); Lg = downstream(model, o[:, :8], o[:, 8:])
            loss = -(pf[a:a + 5000] * torch.log_softmax(Lg, 1)).sum(1).mean() + lam * ((o - Oft[a:a + 5000]) ** 2 / vo).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9 or ep == epochs - 1:
            Wn = Wof().detach(); ag = agree(Wn); path.append((ep + 1, ag))
            if ag > best[0]: best = (ag, Wn.cpu().numpy(), ep + 1)
            if ep % 50 == 49: log(f'  S14 epoch {ep + 1}: {100 * ag:.2f}%, {time.time() - t0:.0f} s')
    np.savez(OUT / f'{tag}1_model.npz', W=best[1], kn=kn, knv=knv, nv=nv, uniform=uni, weights='part', basis='extended', choice=choice)
    r = dict(tag=tag, start=a0, one_term_ls=a1, best=best[0], best_epoch=best[2], terms=nterm, terms_start=int((W0[:, :6 * nv] != 0).sum()), path=path, seconds=time.time() - t0)
    (OUT / f'{tag}1_one_term.json').write_text(json.dumps(r, indent=1)); log(f'S14 {tag}1: {100 * best[0]:.2f}% with one term per input per neuron ({nterm} terms; start {r["terms_start"]})'); return r


if __name__ == '__main__':
    a = sys.argv[1:]; run(a[0] if a else 'S5rpc', int(a[1]) if len(a) > 1 else 150)
