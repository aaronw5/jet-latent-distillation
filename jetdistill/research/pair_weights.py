"""B1 — the physics in ParT's pair-interaction weights. ParT adds to every particle-attention logit a learned bias
U_h(i, j) = f_h(ln kT, ln z, ln ΔR, ln m²) (one function per head, the same in all 8 particle blocks). On real jets:
  - the size of U_h against the content part q·k/√d of the attention logits (block 1 and block 8)
  - how simple each f_h is: R² of the best single input, of an additive fit (hinge terms per input), of additive +
    pairwise products
  - each head's shape: the additive components (binned), to read off which splittings the head favours

  python -m jetdistill.research.pair_weights [n_jets]"""
import json, sys, pathlib, time
import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[2] / 'research' / 'results'
NAMES = ['ln kT', 'ln z', 'ln ΔR', 'ln m²']


def capture(n_jets=3000, device=None):
    """the pair inputs (4, pairs), U (8, pairs) of the real pairs i<j, and the content logits of blocks 1 and 8 on them"""
    import torch
    from weaver.nn.model.ParticleTransformer import pairwise_lv_fts
    from ..part.network import ParTNetwork, inputs
    from ..part.data import read_root
    from ..config import DATA
    import os; device = device or os.environ.get('JETDISTILL_DEVICE', 'cpu')
    files = json.loads((DATA / 'train' / 'files.json').read_text())
    J = read_root(files[3]['file']); J = {k: v[:n_jets] for k, v in J.items()}
    net = ParTNetwork('full'); m = net.model; cap = {}
    hs = [m.blocks[b].register_forward_pre_hook(lambda mod, a, kw, b=b: cap.__setitem__(b, (a[0].detach(), kw['attn_mask'].detach())), with_kwargs=True) for b in (0, 7)]
    out = dict(F=[], U=[], C1=[], C8=[])
    order = np.argsort(J['mask'].sum(1))
    with torch.no_grad():
        for a in range(0, n_jets, 200):
            idx = order[a:a + 200]; x, v, mk = inputs({k: q[idx] for k, q in J.items()}, 'full'); T = lambda q: torch.from_numpy(q).to(device)
            m(T(x), T(v), T(mk)); P = cap[0][0].shape[0]; Nb = len(idx); ok = T(mk)[:, 0, :P] > 0
            vv = T(v)[:, :, :P]; i, j = torch.triu_indices(P, P, 1, device=device)
            F = pairwise_lv_fts(vv[:, :, i], vv[:, :, j])                     # (N, 4, pairs): ln kT, ln z, ln ΔR, ln m²
            sel = ok[:, i] & ok[:, j]
            for b, key in ((0, 'C1'), (7, 'C8')):
                xin, am = cap[b]; blk = m.blocks[b]; u = blk.pre_attn_norm(xin)                 # (P, N, 128)
                W, bias = blk.attn.in_proj_weight, blk.attn.in_proj_bias; q = u @ W[:128].T + bias[:128]; k = u @ W[128:256].T + bias[128:256]
                q = q.view(P, Nb, 8, 16).permute(1, 2, 0, 3); k = k.view(P, Nb, 8, 16).permute(1, 2, 0, 3)
                lg = (q @ k.transpose(-1, -2)) / 4.0                          # (N, 8, P, P) content logits
                out[key].append(lg[:, :, i, j].permute(1, 0, 2)[:, sel].cpu().numpy())
                if b == 0: out['U'].append(am.view(Nb, 8, P, P)[:, :, i, j].permute(1, 0, 2)[:, sel].cpu().numpy())
            out['F'].append(F.permute(1, 0, 2)[:, sel].cpu().numpy())
    for h in hs: h.remove()
    return {k: np.concatenate(v, 1) for k, v in out.items()}


def hinge(v, kn): return np.stack([v] + [np.maximum(0, v - t) for t in kn] + [np.maximum(0, t - v) for t in kn], 1)


def r2(X, y):
    X = np.concatenate([np.ones((len(X), 1)), X], 1); c = np.linalg.lstsq(X, y, rcond=None)[0]; return 1 - ((y - X @ c) ** 2).mean() / y.var(), c


def analyse(D, n_fit=400000):
    rng = np.random.default_rng(0); idx = rng.choice(D['F'].shape[1], min(n_fit, D['F'].shape[1]), replace=False)
    F, U = D['F'][:, idx].T.astype(np.float64), D['U'][:, idx].astype(np.float64)
    kn = [np.quantile(F[:, f], np.linspace(.05, .95, 12)) for f in range(4)]; H = [hinge(F[:, f], kn[f]) for f in range(4)]
    res = []
    for h in range(8):
        y = U[h]; single = [r2(H[f], y)[0] for f in range(4)]; add, c = r2(np.concatenate(H, 1), y)
        prods = np.stack([F[:, a] * F[:, b] for a in range(4) for b in range(a + 1, 4)], 1); add2 = r2(np.concatenate(H + [prods], 1), y)[0]
        # additive components, binned (each input's part of the additive fit, minus its mean)
        comps, s = {}, 1
        for f in range(4):
            w = H[f].shape[1]; part = H[f] @ c[s:s + w]; s += w; part -= part.mean()
            q = np.quantile(F[:, f], np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, F[:, f]) - 1, 0, 9)
            comps[NAMES[f]] = [(float((q[i] + q[i + 1]) / 2), float(part[b == i].mean())) for i in range(10)]
        res.append(dict(head=h + 1, std=float(y.std()), mean=float(y.mean()), r2_single=dict(zip(NAMES, map(float, single))), r2_additive=float(add),
                        r2_additive_products=float(add2), components=comps,
                        content_std_block1=float(D['C1'][h, idx].std()), content_std_block8=float(D['C8'][h, idx].std()),
                        corr_U_content1=float(np.corrcoef(y, D['C1'][h, idx])[0, 1])))
    return res


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000; t0 = time.time(); D = capture(n); R = analyse(D)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'B1_pair_weights.json').write_text(json.dumps(R, indent=1))
    print(f'{D["F"].shape[1]} real pairs of {n} jets, {time.time() - t0:.0f} s')
    for r in R:
        print(f"head {r['head']}: U mean {r['mean']:+.2f} sd {r['std']:.2f} | content logit sd: block 1 {r['content_std_block1']:.2f}, block 8 {r['content_std_block8']:.2f} | "
              f"R² single {max(r['r2_single'].values()):.3f} ({max(r['r2_single'], key=r['r2_single'].get)}), additive {r['r2_additive']:.3f}, +products {r['r2_additive_products']:.3f}")
        for k, v in r['components'].items(): print(f"    {k:6s} " + ' '.join(f'{x:+.1f}:{y:+.2f}' for x, y in v))
