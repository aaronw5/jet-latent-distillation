"""S13 — the pruned per-head model re-tuned against cancellation. For every value neuron, the contribution of each input
f over a batch of jets C_f = Σ_terms of f (pooled term × coefficient); penalty = Σ_f sd(C_f) − sd(Σ_f C_f), relative
to sd(Σ_f C_f): zero when the inputs reinforce each other, large when correlated inputs offset each other. Tuned
(whitened per head, as heads_prune) toward ParT's probabilities + λ_c · penalty; the ratios before/after reported:
  inside  = Σ_f sd(C_f) / Σ_terms sd(term contribution)   (1: an input's hinge pieces do not offset)
  between = sd(Σ_f C_f) / Σ_f sd(C_f)                       (1: inputs do not offset)
Saves {tag}c_model.npz.

  python -m jetdistill.research.heads_decancel TAG [lam_c epochs]"""
import json, sys, time
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import extract, downstream, phi_pooled, rows_of_split, OUT
from .heads_eval import uniformize


def ratios(P, W, nv):
    """median inside / between ratios over all (head, neuron) with live terms"""
    ins, bet = [], []
    for h in range(16):
        for o in range(16):
            C = P[:, h, :6 * nv] * W[h][:6 * nv, o]
            if C.std(0).sum() == 0: continue
            per = np.stack([C[:, [f] + [nv + k * nv + f for k in range(5)]].sum(1) for f in range(nv)], 1)
            ins.append(per.std(0).sum() / C.std(0).sum()); bet.append(per.sum(1).std() / max(per.std(0).sum(), 1e-12))
    return float(np.median(ins)), float(np.median(bet))


def run(tag='S5rp', lam_c=0.05, epochs=200, n_fit=100000, n_dev=20000, lr=3e-4, lam=0.01, device='mps', log=print):
    import torch
    t0 = time.time(); Pz = dict(np.load(OUT / f'{tag}_model.npz')); W0, kn, uni = Pz['W'], Pz['kn'], str(Pz['uniform']); blocks = [int(c) - 1 for c in uni]
    model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    Of, Af, Mf, Lf, _ = extract(model, 'fit', n_fit, device); _, Ad, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Af, Ad = uniformize(Af, Mf, blocks), uniformize(Ad, Md, blocks); Jf, Jd = jets('full', 'fit'), jets('full', 'dev')
    Pf, _ = phi_pooled(Jf, rows_of_split('fit', n_fit), Af, kn); Pd, _ = phi_pooled(Jd, rows_of_split('dev', n_dev), Ad, kn)
    K = Pf.shape[2]; nv = (K - 2) // 6; mask = (W0 != 0).any(2).astype(np.float32)                          # (16, K) live terms (pruning kept)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device); Pft, Pdt = T(Pf), T(Pd); pf = torch.softmax(T(Lf), 1)
    Oft = T(Of.transpose(1, 0, 2, 3).reshape(n_fit, 16, 16)); vo = Oft.var(0) + 1e-6
    G = torch.zeros(K - 2, nv, device=device)                                                                  # term → input map
    for f in range(nv):
        for j in [f] + [nv + k * nv + f for k in range(5)]: G[j, f] = 1
    def agree(W):
        with torch.no_grad():
            return float((torch.cat([downstream(model, *torch.einsum('nhk,hko->nho', Pdt[a:a + 5000], W).split(8, 1)).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
    r0 = ratios(Pd, W0, nv); a0 = agree(T(W0)); log(f'  {tag}: {100 * a0:.2f}%, cancellation ratios inside {r0[0]:.2f} / between {r0[1]:.2f}')
    Ms, Us = [], []
    for h in range(16):
        B = Pf[:, h] * mask[h]; Gm = B.T.astype(np.float64) @ B / len(B); d = np.sqrt(np.maximum(np.diag(Gm), 1e-12))
        ev, Q = np.linalg.eigh(Gm / d[:, None] / d[None]); ev = np.maximum(ev, 1e-3 * ev.max())
        Ms.append((Q / np.sqrt(ev)[None] / d[:, None]) * mask[h][:, None]); Us.append(np.sqrt(ev)[:, None] * (Q.T @ (d[:, None] * W0[h].astype(np.float64))))
    Mt = T(np.stack(Ms)); U = torch.nn.Parameter(T(np.stack(Us))); opt = torch.optim.Adam([U], lr); Wof = lambda: torch.einsum('hkj,hjo->hko', Mt, U)
    best = (a0, T(W0), r0, 0); path = [(0, a0, *r0)]
    for ep in range(epochs):
        for a in np.random.default_rng(ep).permutation(np.arange(0, n_fit, 5000)):
            W = Wof(); Pb = Pft[a:a + 5000]; o = torch.einsum('nhk,hko->nho', Pb, W); Lg = downstream(model, o[:, :8], o[:, 8:])
            loss = -(pf[a:a + 5000] * torch.log_softmax(Lg, 1)).sum(1).mean() + lam * ((o - Oft[a:a + 5000]) ** 2 / vo).mean()
            C = torch.einsum('nhk,hko,kf->nhfo', Pb[:, :, :K - 2], W[:, :K - 2], G)                              # per input f: its contribution to every neuron
            sd_in = C.std(0).sum(1); sd_net = C.sum(2).std(0)                                                   # (16, 16)
            loss = loss + lam_c * ((sd_in - sd_net) / (sd_net + 1e-3)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 19 or ep == epochs - 1:
            Wn = Wof().detach(); ag = agree(Wn); rr = ratios(Pd, Wn.cpu().numpy(), nv); path.append((ep + 1, ag, *rr))
            log(f'  S13 epoch {ep + 1}: {100 * ag:.2f}%, ratios inside {rr[0]:.2f} / between {rr[1]:.2f}, {time.time() - t0:.0f} s')
            if ag >= a0 - 0.002 and rr[1] > best[2][1]: best = (ag, Wn.clone(), rr, ep + 1)                     # keep the least-cancelling model within 0.2 pt
    np.savez(OUT / f'{tag}c_model.npz', W=best[1].cpu().numpy(), kn=kn, uniform=uni, weights='part')
    r = dict(tag=tag, lam_c=lam_c, start=a0, start_ratios=r0, best=best[0], best_ratios=best[2], best_epoch=best[3], path=path, seconds=time.time() - t0)
    (OUT / f'{tag}c_decancel.json').write_text(json.dumps(r, indent=1)); log(f'S13 {tag}c: {100 * best[0]:.2f}%, ratios inside {best[2][0]:.2f} / between {best[2][1]:.2f} (epoch {best[3]})'); return r


if __name__ == '__main__':
    a = sys.argv[1:]; run(a[0] if a else 'S5rp', float(a[1]) if len(a) > 1 else 0.05, int(a[2]) if len(a) > 2 else 200)
