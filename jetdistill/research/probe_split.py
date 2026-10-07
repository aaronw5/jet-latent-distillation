"""A12 — which side of the class attention needs the context: at each level of k-hop physics (A8's inputs, ridge on hinge
terms), ParT's class attention run with (a) only the VALUES predicted (ParT's keys → ParT's α), (b) only the KEYS predicted
(ParT's values), (c) both. Agreement with ParT on the dev jets.

  python -m jetdistill.research.probe_split [n_fit n_dev]"""
import json, sys, time
import numpy as np
from ..config import RESULTS
from ..pipeline import jets
from ..part.clsfit import particle_features
from ..part.network import ParTNetwork
from .nbr import nbr_features, pk_features
from .heads import rows_of_split, OUT
from .heads_attn import kv_true, forward
from .probe_kv import hops_of


def run(n_fit=20000, n_dev=10000, device='mps', log=print):
    import torch
    t0 = time.time(); model = ParTNetwork('full').model; model.eval(); D = RESULTS / '_cls' / 'full'
    rf, rd = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev); Jf, Jd = jets('full', 'fit'), jets('full', 'dev')
    Xf, Mf = np.load(D / 'fit' / 'x_f16.npy', mmap_mode='r')[:n_fit], np.load(D / 'fit' / 'mask.npy')[:n_fit]; Xd, Md, Ld = np.load(D / 'dev' / 'x_f16.npy', mmap_mode='r')[rd], np.load(D / 'dev' / 'mask.npy')[rd], np.load(D / 'dev' / 'L.npy')[rd]
    KVf, KVd = kv_true(model, Xf, Mf, device), kv_true(model, Xd, Md, device); ref = Ld.argmax(1); sets = {}
    for which, J, rows in (('fit', Jf, rf), ('dev', Jd, rd)):
        F0, ok = particle_features(J, rows, ctx=[]); Fn = nbr_features(J['x'][rows], J['ext'][rows], J['jet'][rows])
        Fk1 = pk_features(J['x'][rows], J['ext'][rows], J['jet'][rows], model, device); Fk2 = pk_features(J['x'][rows], J['ext'][rows], J['jet'][rows], model, device, hops=True)[..., 88:]; Fh, _ = hops_of(J, rows, model, device)
        sets[which] = dict(ok=ok, own=np.concatenate([F0, Fn], -1), k1=Fk1, k2=Fk2, hops=Fh)
    okf, okd = sets['fit']['ok'], sets['dev']['ok']; Yf, Yd = KVf[okf].astype(np.float32), KVd[okd].astype(np.float32); T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device, dt)
    def hinge(F, kn, mu, sd): return np.concatenate([(F - mu) / sd] + [np.maximum(0, F - k) / sd for k in kn], -1).astype(np.float32)
    def agree(KVx):
        with torch.no_grad():
            o = np.argsort(okd.sum(1)); predc = np.empty(n_dev, int)
            for a in range(0, n_dev, 2000):
                i = o[a:a + 2000]; P = int(okd[i].sum(1).max()); predc[i] = forward(model, T(KVx[i, :P]), T(okd[i, :P], torch.bool)).argmax(1).cpu().numpy()
        return float((predc == ref).mean())
    KVd32 = KVd.astype(np.float32); res = dict(n_fit=n_fit, n_dev=n_dev, part=agree(KVd32), levels={}); log(f'  ParT keys and values: {100 * res["part"]:.2f}%')
    kcols = np.r_[0:128, 256:384]; vcols = np.r_[128:256, 384:512]; cum_f, cum_d = [], []
    for name in ('own', 'k1', 'k2', 'hops'):
        cum_f.append(sets['fit'][name][okf]); cum_d.append(sets['dev'][name][okd]); Ff, Fd = np.concatenate(cum_f, -1), np.concatenate(cum_d, -1)
        kn = np.quantile(Ff[::7], np.linspace(.15, .85, 5), axis=0).astype(np.float32); mu, sd = Ff.mean(0), Ff.std(0) + 1e-6
        Bf, Bd = hinge(Ff, kn, mu, sd), hinge(Fd, kn, mu, sd); G = Bf.T.astype(np.float64) @ Bf / len(Bf); d = np.sqrt(np.maximum(np.diag(G), 1e-12))
        W = np.linalg.solve(G / d[:, None] / d[None] + 1e-3 * np.eye(len(d)), (Bf.T.astype(np.float64) @ Yf / len(Bf)) / d[:, None]) / d[:, None]
        c0 = Yf.mean(0) - (Bf.mean(0).astype(np.float64) @ W); pred = Bd @ W.astype(np.float32) + c0
        out = {}
        for lab, cols in (('values only', vcols), ('keys only', kcols), ('both', np.r_[0:512])):
            KVp = KVd32.copy(); tmp = KVp[okd]; tmp[:, cols] = pred[:, cols]; KVp[okd] = tmp; out[lab] = agree(KVp)
        res['levels'][name] = dict(inputs=int(Ff.shape[1]), **{k.replace(' ', '_'): v for k, v in out.items()})
        log(f'  up to {name:4s} ({Ff.shape[1]} inputs): predicted values only (ParT’s α) {100 * out["values only"]:.2f}%, keys only (ParT’s values) {100 * out["keys only"]:.2f}%, both {100 * out["both"]:.2f}%, {time.time() - t0:.0f} s')
    res['seconds'] = time.time() - t0; (OUT / 'A12_probe_split.json').write_text(json.dumps(res, indent=1)); return res


if __name__ == '__main__':
    run(*(int(v) for v in sys.argv[1:]))
