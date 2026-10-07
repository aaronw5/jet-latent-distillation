"""S6 — ParT's class attention with formula keys and values (and ParT's own queries). For each particle, formulas give
its key and value for every head of both class blocks (2 × (128 + 128) numbers) from hinge terms of its physics
(own inputs, neighbourhood, ParT pair-kernel context). ParT's own weights do the rest: block 1's query is the
constant class token, block 2's query is ParT's query projection of the (formula) class token after block 1; the class
token's own key/value come from ParT's projections; softmax over [class token, particles]; exact downstream.
Least squares to ParT's true keys/values, then tuned toward ParT's probabilities.

  python -m jetdistill.research.heads_attn [n_fit n_dev epochs]"""
import json, sys, time, pathlib
import numpy as np
from ..config import RESULTS
from ..pipeline import jets
from ..part.clsfit import particle_features
from ..part.network import ParTNetwork
from .nbr import nbr_features, pk_features
from .heads import after

OUT = pathlib.Path(__file__).resolve().parents[2] / 'research' / 'results'


def kv_true(model, X, M, device='mps'):
    """ParT's keys and values of the particles in both class blocks: (N, 128, 512) float16 [k1, v1, k2, v2]"""
    import torch
    N = len(M); out = np.zeros((N, 128, 512), np.float16)
    with torch.no_grad():
        for a in range(0, N, 2000):
            x = torch.from_numpy(np.asarray(X[a:a + 2000], np.float32)).to(device); parts = []
            for blk in model.cls_blocks:
                u = blk.pre_attn_norm(x); W, b = blk.attn.in_proj_weight, blk.attn.in_proj_bias
                parts += [u @ W[128:256].T + b[128:256], u @ W[256:].T + b[256:]]
            out[a:a + 2000] = (torch.cat(parts, -1) * torch.from_numpy(M[a:a + 2000]).to(device)[..., None]).cpu().numpy()
    return out


def feats(J, n, model):
    F, ok = particle_features(J, np.arange(n), ctx=[])
    F = np.concatenate([F, nbr_features(J['x'][:n], J['ext'][:n], J['jet'][:n]), pk_features(J['x'][:n], J['ext'][:n], J['jet'][:n], model)], -1)
    return F, ok


def forward(model, KV, m):
    """class scores from the particles' keys/values KV (n, P, 512) and mask m (n, P), by ParT's own weights"""
    import torch
    n = KV.shape[0]; cls = model.cls_token[0, 0].expand(n, -1); okm = torch.cat([torch.ones_like(m[:, :1]), m], 1)
    for b, blk in enumerate(model.cls_blocks):
        W, bi = blk.attn.in_proj_weight, blk.attn.in_proj_bias; uc = blk.pre_attn_norm(cls)
        q = (cls @ W[:128].T + bi[:128]).view(n, 8, 16); kc = (uc @ W[128:256].T + bi[128:256]).view(n, 1, 8, 16); vc = (uc @ W[256:].T + bi[256:]).view(n, 1, 8, 16)
        k = torch.cat([kc, KV[..., 256 * b:256 * b + 128].view(n, -1, 8, 16)], 1); v = torch.cat([vc, KV[..., 256 * b + 128:256 * b + 256].view(n, -1, 8, 16)], 1)
        s = torch.einsum('nhd,nphd->nhp', q, k) / 4.0; a = torch.softmax(s.masked_fill(~okm[:, None], -float('inf')), -1)
        o = torch.einsum('nhp,nphd->nhd', a, v); cls = after(blk, o, cls)
    return model.fc(model.norm(cls))


def run(n_fit=40000, n_dev=20000, epochs=40, lr=1e-3, lam=0.1, bs=2000, device='mps', log=print):
    import torch
    t0 = time.time(); model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    D = RESULTS / '_cls' / 'full'
    Xf, Mf, Lf = np.load(D / 'fit' / 'x_f16.npy', mmap_mode='r')[:n_fit], np.load(D / 'fit' / 'mask.npy')[:n_fit], np.load(D / 'fit' / 'L.npy')[:n_fit]
    Xd, Md, Ld = np.load(D / 'dev' / 'x_f16.npy', mmap_mode='r')[:n_dev], np.load(D / 'dev' / 'mask.npy')[:n_dev], np.load(D / 'dev' / 'L.npy')[:n_dev]
    KVf, KVd = kv_true(model, Xf, Mf, device), kv_true(model, Xd, Md, device); ref = Ld.argmax(1)
    T = lambda q, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(q)).to(device, dt)
    def agree(kv_of, Mx, nn):
        with torch.no_grad():
            o = np.argsort(Mx.sum(1)); pred = np.empty(nn, int)
            for a in range(0, nn, bs):
                i = o[a:a + bs]; P = int(Mx[i].sum(1).max()); pred[i] = forward(model, kv_of(i, P), T(Mx[i, :P])).argmax(1).cpu().numpy()
        return float((pred == ref).mean())
    chk = agree(lambda i, P: T(KVd[i, :P]), Md, n_dev); log(f'  ParT\'s own keys/values through this forward: {100 * chk:.2f}% (must be 100), {time.time() - t0:.0f} s')
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); Ff, okf = feats(Jf, n_fit, model); Fd, okd = feats(Jd, n_dev, model)
    assert (okf == Mf).all() and (okd == Md).all()
    kn = np.quantile(Ff[okf][::7], np.linspace(.15, .85, 5), axis=0).astype(np.float32); mu, sd = Ff[okf].mean(0), Ff[okf].std(0) + 1e-6
    def phi(F):                                                              # per particle: 1, features (standardized), hinges
        import torch as t
        Fs = (F - T(mu)) / T(sd); ks = (T(kn) - T(mu)) / T(sd)
        return t.cat([t.ones_like(Fs[..., :1]), Fs] + [t.clamp(Fs - ks[j], min=0) for j in range(len(kn))], -1)
    # least squares on a sample of particles
    rs = np.random.default_rng(0); sel = np.argwhere(okf); sel = sel[rs.choice(len(sel), min(400000, len(sel)), replace=False)]
    with torch.no_grad():
        B = phi(T(Ff[sel[:, 0], sel[:, 1]])).double().cpu().numpy(); Y = KVf[sel[:, 0], sel[:, 1]].astype(np.float64)
    G = B.T @ B; d = np.sqrt(np.maximum(np.diag(G), 1e-12)); W0 = np.linalg.solve(G / d[:, None] / d[None] + 1e-6 * np.eye(len(d)), (B.T @ Y) / d[:, None]) / d[:, None]
    r2 = 1 - ((B @ W0 - Y) ** 2).mean(0) / Y.var(0).clip(1e-9); del B, Y
    log(f'  {W0.shape[0]} terms per particle; key/value R² (fit particles): block 1 keys {np.median(r2[:128]):.3f}, values {np.median(r2[128:256]):.3f}, '
        f'block 2 keys {np.median(r2[256:384]):.3f}, values {np.median(r2[384:]):.3f} (medians), {time.time() - t0:.0f} s')
    W = torch.nn.Parameter(T(W0)); Ffh, Fdh = T(Ff, torch.float16), T(Fd, torch.float16); del Ff, Fd
    kv_f = lambda i, P: phi(Ffh[i, :P].float()) @ W * T(Mf[i, :P])[..., None]
    kv_d = lambda i, P: phi(Fdh[i, :P].float()) @ W * T(Md[i, :P])[..., None]
    a0 = agree(kv_d, Md, n_dev); log(f'  least squares: same class as ParT {100 * a0:.2f}%, {time.time() - t0:.0f} s')
    opt = torch.optim.Adam([W], lr); pf = torch.softmax(T(Lf), 1); best = (a0, 0); path = [(0, a0)]; o = np.argsort(Mf.sum(1))
    for ep in range(epochs):
        for a in np.random.default_rng(ep).permutation(np.arange(0, n_fit, bs)):
            i = o[a:a + bs]; P = int(Mf[i].sum(1).max()); kv = kv_f(i, P)
            loss = -(pf[i] * torch.log_softmax(forward(model, kv, T(Mf[i, :P])), 1)).sum(1).mean() + lam * ((kv - T(KVf[i, :P])) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if device == 'mps': torch.mps.empty_cache()
        if ep % 5 == 4 or ep == epochs - 1:
            ag = agree(kv_d, Md, n_dev); path.append((ep + 1, ag)); best = max(best, (ag, ep + 1))
            log(f'  S6 epoch {ep + 1}: same class as ParT {100 * ag:.2f}%, {time.time() - t0:.0f} s')
    return dict(experiment='S6', n_fit=n_fit, n_dev=n_dev, terms=int(W0.shape[0]), check=chk, least_squares=a0, best=best[0], best_epoch=best[1], path=path, lr=lr, lam=lam,
                kv_r2=dict(k1=float(np.median(r2[:128])), v1=float(np.median(r2[128:256])), k2=float(np.median(r2[256:384])), v2=float(np.median(r2[384:]))), seconds=time.time() - t0)


if __name__ == '__main__':
    a = [int(v) for v in sys.argv[1:]]; r = run(*a); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'S6_heads_attn.json').write_text(json.dumps(r, indent=1)); print('RESULT', json.dumps(r))
