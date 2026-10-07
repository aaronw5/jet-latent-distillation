"""A8 proxy — how much of ParT's per-particle keys and values (what the class attention reads) can k-hop physics
context explain? Targets: the block-1 and block-2 keys and values of every real particle (as heads_attn.kv_true).
Inputs, cumulative: own (18 + 20 neighbourhood), + 1 hop of ParT-kernel context (88), + 2nd hop and pT-weighted (A1),
+ learned-free simple hops: the kernel-averaged *own inputs* of the neighbours repeated 2 and 3 times (h ← mean_w(h)).
Ridge regression on hinge terms; R² per target on held-out jets; plus the agreement when the predicted keys/values
replace ParT's in the class attention (heads_attn.forward).

  python -m jetdistill.research.probe_kv [n_fit n_dev]"""
import json, sys, time
import numpy as np
from ..config import RESULTS
from ..pipeline import jets
from ..part.clsfit import particle_features
from ..part.network import ParTNetwork
from .nbr import nbr_features, pk_features
from .heads import rows_of_split, OUT
from .heads_attn import kv_true, forward


def hops_of(J, rows, model, device, nh=3, chunk=250):
    """kernel-averaged own inputs repeated: h¹ = Σ_j w_ij x_j (per kernel head, mean over the 8 kernels), h² = Σ_j w_ij h¹_j, …"""
    import torch
    x, ext, jet = J['x'][rows], J['ext'][rows], J['jet'][rows]; F0, ok = particle_features(J, rows, ctx=[]); emb = model.pair_embed.embed
    out = np.zeros(F0.shape[:2] + (nh, 18), np.float32)
    for a in range(0, len(rows), chunk):
        xx = np.asarray(x[a:a + chunk], np.float32); okc = ok[a:a + chunk]; P = int(okc.sum(1).max()); n = len(xx)
        T = lambda q: torch.from_numpy(np.ascontiguousarray(q)).to(device)
        pt, eta, phi = T(xx[:, :P, 0]), T(xx[:, :P, 1]), T(xx[:, :P, 2]); m = T(okc[:, :P])
        with torch.no_grad():
            deta = eta[:, :, None] - eta[:, None]; dphi = phi[:, :, None] - phi[:, None]; dR = torch.sqrt(deta ** 2 + dphi ** 2).clamp(min=1e-8)
            ptmin = torch.minimum(pt[:, :, None], pt[:, None]).clamp(min=1e-8)
            Fp = torch.stack([torch.log((ptmin * dR).clamp(min=1e-8)), torch.log((ptmin / (pt[:, :, None] + pt[:, None]).clamp(min=1e-8)).clamp(min=1e-8)), torch.log(dR),
                              torch.log((2 * pt[:, :, None] * pt[:, None] * (torch.cosh(deta) - torch.cos(dphi))).clamp(min=1e-8))], 1).reshape(n, 4, P * P)
            U = emb(Fp).reshape(n, 8, P, P).mean(1)                                                          # the 8 kernels averaged: one neighbourhood weighting
            pair = m[:, :, None] & m[:, None] & ~torch.eye(P, dtype=torch.bool, device=device)[None]
            w = torch.softmax(U.masked_fill(~pair, -1e9), -1) * m[:, None, :]
            h = T(F0[a:a + chunk, :P, :18]) * m[..., None]
            for k in range(nh):
                h = torch.einsum('nij,njf->nif', w, h) * m[..., None]; out[a:a + chunk, :P, k] = h.cpu().numpy()
    return out.reshape(F0.shape[0], F0.shape[1], -1), ok


def run(n_fit=20000, n_dev=10000, device='mps', log=print):
    import torch
    t0 = time.time(); model = ParTNetwork('full').model; model.eval(); D = RESULTS / '_cls' / 'full'
    rf, rd = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev); Jf, Jd = jets('full', 'fit'), jets('full', 'dev')
    Xf, Mf = np.load(D / 'fit' / 'x_f16.npy', mmap_mode='r')[:n_fit], np.load(D / 'fit' / 'mask.npy')[:n_fit]; Xd, Md, Ld = np.load(D / 'dev' / 'x_f16.npy', mmap_mode='r')[rd], np.load(D / 'dev' / 'mask.npy')[rd], np.load(D / 'dev' / 'L.npy')[rd]
    KVf, KVd = kv_true(model, Xf, Mf, device), kv_true(model, Xd, Md, device); ref = Ld.argmax(1)
    sets = {}
    for which, J, rows in (('fit', Jf, rf), ('dev', Jd, rd)):
        F0, ok = particle_features(J, rows, ctx=[]); Fn = nbr_features(J['x'][rows], J['ext'][rows], J['jet'][rows])
        Fk1 = pk_features(J['x'][rows], J['ext'][rows], J['jet'][rows], model, device); Fk2 = pk_features(J['x'][rows], J['ext'][rows], J['jet'][rows], model, device, hops=True)[..., 88:]
        Fh, _ = hops_of(J, rows, model, device)
        sets[which] = dict(ok=ok, own=np.concatenate([F0, Fn], -1), k1=Fk1, k2=Fk2, hops=Fh)
        log(f'  {which}: inputs built, {time.time() - t0:.0f} s')
    okf, okd = sets['fit']['ok'], sets['dev']['ok']; Yf, Yd = KVf[okf].astype(np.float32), KVd[okd].astype(np.float32)
    res = dict(n_fit=n_fit, n_dev=n_dev, levels={})
    def hinge(F, kn, mu, sd): return np.concatenate([(F - mu) / sd] + [np.maximum(0, F - k) / sd for k in kn], -1).astype(np.float32)
    cum_f, cum_d = [], []
    for name in ('own', 'k1', 'k2', 'hops'):
        cum_f.append(sets['fit'][name][okf]); cum_d.append(sets['dev'][name][okd]); Ff, Fd = np.concatenate(cum_f, -1), np.concatenate(cum_d, -1)
        kn = np.quantile(Ff[::7], np.linspace(.15, .85, 5), axis=0).astype(np.float32); mu, sd = Ff.mean(0), Ff.std(0) + 1e-6
        Bf, Bd = hinge(Ff, kn, mu, sd), hinge(Fd, kn, mu, sd); G = Bf.T.astype(np.float64) @ Bf / len(Bf); d = np.sqrt(np.maximum(np.diag(G), 1e-12))
        W = np.linalg.solve(G / d[:, None] / d[None] + 1e-3 * np.eye(len(d)), (Bf.T.astype(np.float64) @ Yf / len(Bf)) / d[:, None]) / d[:, None]
        c0 = Yf.mean(0) - (Bf.mean(0).astype(np.float64) @ W); pred = Bd @ W.astype(np.float32) + c0; r2 = 1 - ((pred - Yd) ** 2).mean(0) / Yd.var(0).clip(1e-9)
        # agreement with the predicted keys and values in the class attention
        KVp = np.zeros_like(KVd, dtype=np.float32); KVp[okd] = pred; T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device, dt)
        with torch.no_grad():
            o = np.argsort(okd.sum(1)); predc = np.empty(n_dev, int)
            for a in range(0, n_dev, 2000):
                i = o[a:a + 2000]; P = int(okd[i].sum(1).max()); predc[i] = forward(model, T(KVp[i, :P]), T(okd[i, :P], torch.bool)).argmax(1).cpu().numpy()
        ag = float((predc == ref).mean())
        res['levels'][name] = dict(inputs=int(Ff.shape[1]), terms=int(Bf.shape[1]), r2=dict(k1=float(np.median(r2[:128])), v1=float(np.median(r2[128:256])), k2=float(np.median(r2[256:384])), v2=float(np.median(r2[384:]))), agreement=ag)
        log(f'  up to {name:4s} ({Ff.shape[1]} inputs): median R² keys/values block 1 {np.median(r2[:128]):.3f}/{np.median(r2[128:256]):.3f}, block 2 {np.median(r2[256:384]):.3f}/{np.median(r2[384:]):.3f}; '
            f'predicted keys+values in the class attention: {100 * ag:.2f}%, {time.time() - t0:.0f} s')
    res['seconds'] = time.time() - t0; return res


if __name__ == '__main__':
    a = [int(v) for v in sys.argv[1:]]; r = run(*a); (OUT / 'A8_probe_kv.json').write_text(json.dumps(r, indent=1)); print('RESULT', json.dumps(r['levels']))
