"""Reproducing ParT component by component: swap one component for a formula version, keep the rest of the real
ParT, measure agreement with ParT (and how close the internal activations stay). Components so far:
  pair    the pair kernel U_h(ln kT, ln z, ln ΔR, ln m²) of each head → an additive hinge formula per head
  embed   the per-particle input embedding (17 inputs → 128) → a hinge formula per output of the inputs
Each formula: least squares to ParT's own component on real particles/pairs, then (optionally) tuned through the
frozen rest of ParT toward ParT's probabilities.

  python -m jetdistill.research.replace dump [n_fit n_dev]    ParT's exact inputs of the fitting / validation jets
  python -m jetdistill.research.replace run <components> [tune_steps]     e.g. none, pair, embed, pair+embed"""
import json, sys, time, pathlib
import numpy as np
from ..config import RESULTS, DATA

OUT = pathlib.Path(__file__).resolve().parents[2] / 'research' / 'results'
D = RESULTS / '_inputs' / 'full'


def dump(n_fit=100000, n_dev=50000, log=print):
    from ..part.data import read_root
    from ..part.network import ParTNetwork, inputs
    from ..part.clsfit import rows_of
    files = json.loads((DATA / 'train' / 'files.json').read_text()); off = np.cumsum([0] + [f['jets'] for f in files]); net = ParTNetwork('full')
    for which, n in (('fit', n_fit), ('dev', n_dev)):
        t0 = time.time(); r = rows_of(which, n)[:n]; d = D / which; d.mkdir(parents=True, exist_ok=True)
        X = np.lib.format.open_memmap(d / 'x.npy', 'w+', np.float16, (len(r), 17, 128)); V = np.lib.format.open_memmap(d / 'v.npy', 'w+', np.float32, (len(r), 4, 128))
        M = np.zeros((len(r), 128), bool); L = np.zeros((len(r), 10), np.float32)
        for fi, f in enumerate(files):
            pos = np.flatnonzero((r >= off[fi]) & (r < off[fi + 1]))
            if not len(pos): continue
            J = read_root(f['file']); J = {k: v[r[pos] - off[fi]] for k, v in J.items()}
            for a in range(0, len(pos), 5000):
                Jc = {k: v[a:a + 5000] for k, v in J.items()}; x, v, m = inputs(Jc, 'full'); P = x.shape[-1]; p = pos[a:a + 5000]
                X[p, :, :P] = x.astype(np.float16); V[p, :, :P] = v; M[p, :P] = m[:, 0] > 0
                L[p] = net.run(Jc)['logits']
            log(f'  {which}: {f["file"]}: {len(pos)} jets, {time.time() - t0:.0f} s')
        X.flush(); V.flush(); np.save(d / 'mask.npy', M); np.save(d / 'L.npy', L)
        log(f'{which}: {len(r)} jets in {time.time() - t0:.0f} s')


def load(which, n=None):
    d = D / which; M = np.load(d / 'mask.npy'); n = n or len(M)
    return np.load(d / 'x.npy', mmap_mode='r')[:n], np.load(d / 'v.npy', mmap_mode='r')[:n], M[:n], np.load(d / 'L.npy')[:n]


def batches(X, V, M, bs=500, device='mps'):
    """ParT's inputs batch by batch (jets of similar multiplicity together, padded slots refilled as weaver does)"""
    import torch
    o = np.argsort(M.sum(1), kind='stable')
    for a in range(0, len(o), bs):
        i = np.sort(o[a:a + bs]); P = int(M[i].sum(1).max()); m = M[i, :P]; cnt = m.sum(1)
        wrap = np.arange(P)[None] % np.maximum(cnt, 1)[:, None]
        x = np.asarray(X[i, :, :P], np.float32); v = np.asarray(V[i, :, :P], np.float32)
        fill = lambda a: np.where(m[:, None], a, np.take_along_axis(a, np.broadcast_to(wrap[:, None], a.shape), 2))
        T = lambda q: torch.from_numpy(np.ascontiguousarray(q)).to(device)
        yield i, T(fill(x)), T(fill(v)), T(m[:, None].astype(np.float32))


# ---------------------------------------------------------------- formula versions of components
def hinge_t(F, knots):
    """(…, K) torch: 1, each input, and its hinges at its thresholds"""
    import torch
    cols = [torch.ones_like(F[..., :1])]
    for f, kn in enumerate(knots):
        v = F[..., f:f + 1]; T = torch.as_tensor(kn, device=F.device, dtype=F.dtype); cols += [v]
        if len(kn): cols += [torch.clamp(v - T, min=0), torch.clamp(T - v, min=0)]
    return torch.cat(cols, -1)


def lsq(B, Y, ridge=1e-6):
    """least squares in float64 on the CPU (column-scaled normal equations)"""
    B = B.double().cpu().numpy() if hasattr(B, 'cpu') else B; Y = Y.double().cpu().numpy() if hasattr(Y, 'cpu') else Y
    G = B.T @ B; b = B.T @ Y; d = np.sqrt(np.maximum(np.diag(G), 1e-12))
    return np.linalg.solve(G / d[:, None] / d[None] + ridge * np.eye(len(d)), b / d[:, None]) / d[:, None]


class PairFormula:
    """U_h ≈ Σ_f g_hf(input f), hinge terms at 12 quantiles of each pair input"""
    def __init__(self, model, X, V, M, n_jets=3000, device='mps'):
        import torch
        self.model, self.device = model, device; Fs, Us = [], []
        cap = {}; hk = model.pair_embed.register_forward_hook(lambda mod, a, out: cap.__setitem__('U', out))
        hk2 = model.pair_embed.embed.register_forward_hook(lambda mod, a, out: cap.__setitem__('F', a[0]))
        with torch.no_grad():
            for i, x, v, m in batches(X[:n_jets], V[:n_jets], M[:n_jets], device=device):
                model(x, v, m); F = cap['F']                                    # F: (N, 4, pairs of the lower triangle)
                Fs.append(F.permute(0, 2, 1).reshape(-1, 4)[::7]); Us.append(model.pair_embed.embed(F).permute(0, 2, 1).reshape(-1, 8)[::7])
        hk.remove(); hk2.remove(); F = torch.cat(Fs); U = torch.cat(Us)
        Fc = F.cpu().numpy(); self.knots = [np.quantile(Fc[:, f], np.linspace(.04, .96, 12)).astype(np.float32) for f in range(4)]
        B = hinge_t(F, self.knots); self.W = torch.from_numpy(lsq(B, U).astype(np.float32)).to(device)
        pred = B @ self.W; self.r2 = (1 - ((pred - U) ** 2).mean(0) / U.var(0)).cpu().numpy()
    def params(self): return [self.W]
    def install(self):
        """the model's pair_embed.embed → the formula (same input (N, 4, pairs), same output (N, 8, pairs))"""
        emb = self.model.pair_embed.embed; self._orig = emb.forward
        emb.forward = lambda F: (hinge_t(F.permute(0, 2, 1), self.knots) @ self.W).permute(0, 2, 1)
    def remove(self): self.model.pair_embed.embed.forward = self._orig


class EmbedFormula:
    """each of the 128 outputs of ParT's input embedding ≈ a hinge formula of the particle's 17 scaled inputs"""
    def __init__(self, model, X, V, M, n_jets=4000, device='mps'):
        import torch
        self.model, self.device = model, device; xs, ys = [], []
        with torch.no_grad():
            for i, x, v, m in batches(X[:n_jets], V[:n_jets], M[:n_jets], device=device):
                y = model.embed(x); xin = x.permute(2, 0, 1)                    # embed: (N, 17, P) -> (P, N, 128)
                ok = m[:, 0].T > 0; xs.append(xin[ok]); ys.append(y[ok])
        xf = torch.cat(xs); y = torch.cat(ys); xc = xf.cpu().numpy()
        self.knots = [np.unique(np.quantile(xc[:, f], np.linspace(.05, .95, 9))).astype(np.float32) if len(np.unique(xc[:, f][:20000])) > 3 else np.zeros(0, np.float32) for f in range(17)]
        B = hinge_t(xf, self.knots); self.W = torch.from_numpy(lsq(B, y).astype(np.float32)).to(device)
        pred = B @ self.W; self.r2 = (1 - ((pred - y) ** 2).mean(0) / y.var(0).clamp(min=1e-9)).cpu().numpy()
    def params(self): return [self.W]
    def install(self):
        emb = self.model.embed; self._orig = emb.forward
        emb.forward = lambda x: hinge_t(x.permute(2, 0, 1), self.knots) @ self.W     # x: (N, 17, P) -> (P, N, 128)
    def remove(self): self.model.embed.forward = self._orig


COMPONENTS = dict(pair=PairFormula, embed=EmbedFormula)


def agreement(model, X, V, M, L, device='mps'):
    import torch
    pred = np.empty(len(M), int)
    with torch.no_grad():
        for i, x, v, m in batches(X, V, M, device=device): pred[i] = model(x, v, m).argmax(1).cpu().numpy()
    return float((pred == L.argmax(1)).mean())


def run(components='none', steps=0, lr=1e-3, n_tune=30000, device='mps', log=print):
    import torch
    from ..part.network import ParTNetwork
    t0 = time.time(); model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    Xf, Vf, Mf, Lf = load('fit'); Xd, Vd, Md, Ld = load('dev')
    res = dict(components=components, steps=steps)
    res['exact'] = a0 = agreement(model, Xd, Vd, Md, Ld); log(f'  ParT itself on the stored inputs: {100 * a0:.2f}% (must be 100)')
    comps = [] if components == 'none' else [COMPONENTS[c](model, Xf, Vf, Mf, device=device) for c in components.split('+')]
    for c, name in zip(comps, components.split('+')):
        res[f'r2_{name}'] = dict(median=float(np.median(c.r2)), min=float(c.r2.min()), mean=float(c.r2.mean())); c.install()
        log(f'  {name}: formula R² to ParT\'s component median {np.median(c.r2):.4f}, min {c.r2.min():.4f}')
    if comps:
        res['least_squares'] = a1 = agreement(model, Xd, Vd, Md, Ld); log(f'  {components} as formulas (least squares), rest of ParT: {100 * a1:.2f}%, {time.time() - t0:.0f} s')
        if steps:
            ps = [p for c in comps for p in c.params()]; [p.requires_grad_(True) for p in ps]; opt = torch.optim.Adam(ps, lr)
            Pf = torch.softmax(torch.from_numpy(Lf).to(device), 1); best = a1; k = 0
            while k < steps:
                for i, x, v, m in batches(Xf[:n_tune], Vf[:n_tune], Mf[:n_tune], bs=256, device=device):
                    loss = -(Pf[i] * torch.log_softmax(model(x, v, m), 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step(); k += 1
                    if k % 100 == 0:
                        a = agreement(model, Xd, Vd, Md, Ld); best = max(best, a); log(f'    tuning step {k}: {100 * a:.2f}%, {time.time() - t0:.0f} s')
                    if k >= steps: break
            res['tuned'] = best
    res['seconds'] = time.time() - t0
    return res


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'dump': dump(*(int(v) for v in a))
    else:
        r = run(a[0], int(a[1]) if len(a) > 1 else 0); OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f'R_{a[0]}_{r["steps"]}.json').write_text(json.dumps(r, indent=1)); print('RESULT', json.dumps(r))
