"""Pruning a per-head model (S5 / S8 / their readable refits): which inputs does each head need? Per (head, input):
its 6 terms zeroed, the agreement drop on the balanced dev jets; the least-needed groups removed while their summed drop
stays below TOL, the rest re-tuned (whitened per head, as heads.run) toward ParT's probabilities; repeated until nothing
can go. Saves research/results/{tag}p_model.npz (the pruned model, same format) and {tag}p_pruned.json (the path).

  python -m jetdistill.research.heads_prune TAG [tol_pt rounds epochs]"""
import json, sys, time
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import extract, downstream, phi_pooled, rows_of_split, OUT
from .heads_eval import uniformize


def run(tag='S5', tol=0.1, rounds=8, epochs=40, n_fit=100000, n_dev=20000, lr=3e-4, lam=0.01, device='mps', log=print):
    import torch
    t0 = time.time(); Pz = dict(np.load(OUT / f'{tag}_model.npz')); W0, kn, uni = Pz['W'], Pz['kn'], str(Pz['uniform']); blocks = [int(c) - 1 for c in uni]
    model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    Of, Af, Mf, Lf, _ = extract(model, 'fit', n_fit, device); _, Ad, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Af, Ad = uniformize(Af, Mf, blocks), uniformize(Ad, Md, blocks); Jf, Jd = jets('full', 'fit'), jets('full', 'dev')
    Pf, _ = phi_pooled(Jf, rows_of_split('fit', n_fit), Af, kn); Pd, _ = phi_pooled(Jd, rows_of_split('dev', n_dev), Ad, kn); log(f'  pooled terms, {time.time() - t0:.0f} s')
    K = Pf.shape[2]; nv = (K - 2) // 6; group = lambda f: [f] + [nv + k * nv + f for k in range(5)]          # the 6 terms of input f
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device); Pft, Pdt = T(Pf), T(Pd); pf = torch.softmax(T(Lf), 1)
    Oft = T(Of.transpose(1, 0, 2, 3).reshape(n_fit, 16, 16)); vo = Oft.var(0) + 1e-6
    def agree(W):
        with torch.no_grad():
            return float((torch.cat([downstream(model, *torch.einsum('nhk,hko->nho', Pdt[a:a + 5000], W).split(8, 1)).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
    W = T(W0); keep = np.ones((16, nv), bool)
    if 'dead' in Pz: keep &= ~np.stack([Pz['dead'][h][:nv] & Pz['dead'][h][nv:2 * nv] for h in range(16)])
    a_ref = a_now = agree(W); path = [dict(inputs=int(keep.sum()), agreement=a_now)]; log(f'  start: {int(keep.sum())} (head, input) pairs, {100 * a_now:.2f}%')
    for rnd in range(rounds):
        drops = {}
        for h in range(16):
            for f in np.flatnonzero(keep[h]):
                Wt = W.clone(); Wt[h, group(f)] = 0; drops[(h, f)] = a_now - agree(Wt)
        order = sorted(drops, key=drops.get); cum, rem = 0.0, []
        for hf in order:
            if cum + max(drops[hf], 0) > tol / 100: break
            cum += max(drops[hf], 0); rem.append(hf)
        if not rem: break
        for h, f in rem: keep[h, f] = False
        mask = np.ones((16, K), np.float32)
        for h in range(16):
            for f in np.flatnonzero(~keep[h]): mask[h, group(f)] = 0
        # re-tune in whitened coordinates per head over the live columns
        Ms, Us = [], []
        Wn = (W * T(mask)[..., None]).cpu().numpy().astype(np.float64)
        for h in range(16):
            B = Pf[:, h] * mask[h]; G = B.T.astype(np.float64) @ B / len(B); d = np.sqrt(np.maximum(np.diag(G), 1e-12))
            ev, Q = np.linalg.eigh(G / d[:, None] / d[None]); ev = np.maximum(ev, 1e-3 * ev.max())
            Ms.append((Q / np.sqrt(ev)[None] / d[:, None]) * mask[h][:, None]); Us.append(np.sqrt(ev)[:, None] * (Q.T @ (d[:, None] * Wn[h])))
        Mt = T(np.stack(Ms)); U = torch.nn.Parameter(T(np.stack(Us))); opt = torch.optim.Adam([U], lr)
        Wof = lambda: torch.einsum('hkj,hjo->hko', Mt, U)
        best = (agree(Wof().detach()), Wof().detach().clone())
        for ep in range(epochs):
            for a in np.random.default_rng(ep).permutation(np.arange(0, n_fit, 5000)):
                o = torch.einsum('nhk,hko->nho', Pft[a:a + 5000], Wof()); Lg = downstream(model, o[:, :8], o[:, 8:])
                loss = -(pf[a:a + 5000] * torch.log_softmax(Lg, 1)).sum(1).mean() + lam * ((o - Oft[a:a + 5000]) ** 2 / vo).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 10 == 9:
                ag = agree(Wof().detach())
                if ag > best[0]: best = (ag, Wof().detach().clone())
        W = best[1] * T(mask)[..., None]; a_now = agree(W)
        path.append(dict(round=rnd + 1, removed=[[int(h), int(f)] for h, f in rem], inputs=int(keep.sum()), agreement=a_now))
        log(f'  round {rnd + 1}: removed {len(rem)} (head, input) pairs → {int(keep.sum())} kept ({keep.sum(1).min()}–{keep.sum(1).max()} inputs per head), re-tuned {100 * a_now:.2f}% (start {100 * a_ref:.2f}%), {time.time() - t0:.0f} s')
        np.savez(OUT / f'{tag}p_model.npz', W=W.cpu().numpy(), kn=kn, uniform=uni, weights='part', keep=keep)
        if a_now < a_ref - tol / 100: log('  below the tolerance: stop'); break
    r = dict(tag=tag, tol=tol, start=a_ref, final=a_now, inputs_per_head=keep.sum(1).tolist(), path=path, seconds=time.time() - t0)
    (OUT / f'{tag}p_pruned.json').write_text(json.dumps(r, indent=1)); log(f'pruned {tag}: {int(keep.sum())} (head, input) pairs, {100 * a_now:.2f}%'); return r


if __name__ == '__main__':
    a = sys.argv[1:]; run(a[0], *(float(v) if i == 0 else int(v) for i, v in enumerate(a[1:])))
