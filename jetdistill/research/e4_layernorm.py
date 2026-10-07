"""E4 — does matching ParT's final LayerNorm help the jet-level fit? ParT: class token c (128) → LayerNorm → h (the 128
"neurons" the formulas fit) → last layer. LN couples the neurons (minus their mean, divided by their spread). Same inputs,
same model class, two targets:
  post   formula → h directly → last layer                    (as now)
  pre    formula → c, then ParT's own LN (γ, β) → last layer  (the norm kept as a fixed operation)
Model class: linear in the quantile-normalized jet quantities, and an additive hinge model of the top-K quantities
(formula-like); least squares to the target, then tuned toward ParT's probabilities (+ λ·R on the target), GPU.

  python -m jetdistill.research.e4_layernorm"""
import json, sys, time, pathlib
import numpy as np
from ..pipeline import jets
from ..config import RESULTS
from .ceiling_jet import matrix

OUT = pathlib.Path(__file__).resolve().parents[2] / 'research' / 'results'


def pre_ln(which, n, device='mps'):
    """ParT's class token before the final LayerNorm, from the extracted particle embeddings (results_part_plus/_cls)"""
    import torch
    from ..part.network import ParTNetwork
    d = RESULTS / '_cls' / 'full' / which; X = np.load(d / 'x_f16.npy', mmap_mode='r'); M = np.load(d / 'mask.npy')
    model = ParTNetwork('full').model; model.eval(); C = np.empty((n, 128), np.float32); L = np.empty((n, 10), np.float32)
    o = np.argsort(M[:n].sum(1))
    with torch.no_grad():
        for a in range(0, n, 2000):
            i = np.sort(o[a:a + 2000]); P = int(M[i].sum(1).max()); m = torch.from_numpy(M[i, :P]).to(device)
            x = torch.from_numpy(np.asarray(X[i, :P], np.float32)).to(device).permute(1, 0, 2); cls = model.cls_token.expand(1, len(i), -1)
            for b in model.cls_blocks: cls = b(x, x_cls=cls, padding_mask=~m)
            C[i] = cls.squeeze(0).cpu().numpy(); L[i] = model.fc(model.norm(cls).squeeze(0)).cpu().numpy()
    return C, L, model


def run(n=100000, K=60, steps=400, lr=3e-3, lam=0.01, device='mps', log=print):
    import torch
    from sklearn.preprocessing import QuantileTransformer
    t0 = time.time(); Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); keys = [k for k in Jf['Q']]
    Xf, Xd = matrix(Jf, keys)[:n], matrix(Jd, keys)[:n]; ok = np.isfinite(Xf).all(0) & (Xf.std(0) > 0); Xf, Xd = Xf[:, ok], Xd[:, ok]
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', subsample=100000, random_state=0).fit(Xf)
    Xf, Xd = np.clip(qt.transform(Xf), -5, 5).astype(np.float32), np.clip(qt.transform(Xd), -5, 5).astype(np.float32)
    Cf, Lf, model = pre_ln('fit', n, device); Cd, Ld, _ = pre_ln('dev', n, device)
    assert (Lf.argmax(1) == Jf['net'][:n]).mean() > .999 and (Ld.argmax(1) == Jd['net'][:n]).mean() > .999, 'embeddings and jets misaligned'
    Hf = np.asarray(Jf['H'][:n], np.float32)
    # additive hinge basis of the K quantities most correlated with the neurons (formula-like)
    cor = np.abs(np.corrcoef(Xf[:20000].T, Hf[:20000].T)[:Xf.shape[1], Xf.shape[1]:]).max(1); top = np.argsort(-cor)[:K]
    kn = np.linspace(-1.5, 1.5, 7)
    hb = lambda X: np.concatenate([X] + [np.maximum(0, X[:, top] - t) for t in kn] + [np.maximum(0, t - X[:, top]) for t in kn], 1).astype(np.float32)
    T = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(device)
    norm, fc = model.norm, model.fc
    for p in model.parameters(): p.requires_grad_(False)
    pf, nd = T(Jf['P'][:n]), Jd['net'][:n]
    res = {}
    for basis_name, bf, bd in (('linear', Xf, Xd), ('hinge', hb(Xf), hb(Xd))):
        Bf, Bd = T(np.concatenate([bf, np.ones((n, 1), np.float32)], 1)), T(np.concatenate([bd, np.ones((n, 1), np.float32)], 1))
        G = (bf.T.astype(np.float64) @ bf); G = np.block([[G, bf.sum(0, dtype=np.float64)[:, None]], [bf.sum(0, dtype=np.float64)[None], np.array([[n]])]])
        for target, Y in (('post', Hf), ('pre', Cf)):
            b = np.concatenate([bf, np.ones((n, 1), np.float32)], 1).T.astype(np.float64) @ Y
            d = np.sqrt(np.diag(G)); W0 = np.linalg.solve(G / d[:, None] / d[None] + 1e-6 * np.eye(len(d)), b / d[:, None]) / d[:, None]
            W = T(W0).requires_grad_(True); Yt = T(Y); vn = Yt.var(0) + 1e-6
            head = (lambda y: fc(y)) if target == 'post' else (lambda y: fc(norm(y)))
            ag = lambda: float((torch.cat([head(Bd[a:a + 20000] @ W).argmax(1) for a in range(0, n, 20000)]).cpu().numpy() == nd).mean())
            with torch.no_grad(): a0 = ag()
            opt = torch.optim.Adam([W], lr); best = a0
            for s in range(steps):
                y = Bf @ W; loss = -(pf * torch.log_softmax(head(y), 1)).sum(1).mean() + lam * ((y - Yt) ** 2 / vn).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                if s % 50 == 49:
                    with torch.no_grad(): best = max(best, ag())
            res[f'{basis_name}/{target}'] = dict(least_squares=a0, tuned=best, n_coef=int(W.numel()))
            log(f'  {basis_name:6s} {target:4s}: least squares {100 * a0:.2f}%, tuned {100 * best:.2f}% ({W.shape[0]} terms per output), {time.time() - t0:.0f} s')
    return dict(experiment='E4', n=n, K=K, steps=steps, lam=lam, results=res, seconds=time.time() - t0)


if __name__ == '__main__':
    r = run(); OUT.mkdir(parents=True, exist_ok=True); (OUT / 'E4_layernorm.json').write_text(json.dumps(r, indent=1)); print('RESULT', json.dumps(r))
