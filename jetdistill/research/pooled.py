"""Step 3 (JEDI spirit) — jet quantities shaped like ParT's class attention (S1): block 2 ≈ a plain mean over the
particles, block 1 ≈ a few selected particles. Per jet:
  mean_i f(i), mean_i max(0, f(i) − t) (5 thresholds), Σ_i z_i f(i)   for every per-particle function f: its 18 own
      inputs, 20 neighbourhood features, 88 ParT pair-kernel context features (z = pT share)
  the own + neighbourhood features of 3 selected particles: the hardest charged, the widest-angle non-soft
      (pT share > 1%), the most displaced charged (largest |d0|/σ)
Saved to RESULTS/_pooled/full/{fit,dev}.npy (float32) with names.json. Then the ceiling check (MLP toward ParT's
probabilities, as E1) on: existing quantities / pooled / both.

  python -m jetdistill.research.pooled build [n_fit] | check"""
import json, sys, time, pathlib
import numpy as np
from ..config import RESULTS
from ..pipeline import jets
from ..part.clsfit import particle_features, PFEAT
from ..part.network import ParTNetwork
from .nbr import nbr_features, pk_features, NBR, PK

OUT = pathlib.Path(__file__).resolve().parents[2] / 'research' / 'results'
DP = RESULTS / '_pooled' / 'full'
NOWN = len(PFEAT)


def per_particle(J, r, model):
    F, ok = particle_features(J, r, ctx=[])                               # own inputs only (no jet context)
    Fn = nbr_features(J['x'][r], J['ext'][r], J['jet'][r]); Fp = pk_features(J['x'][r], J['ext'][r], J['jet'][r], model)
    return np.concatenate([F, Fn, Fp], -1), ok


def pool(F, ok, J, r, kn):
    n = ok.sum(1, keepdims=True).clip(1); okf = ok[..., None]
    mean = (F * okf).sum(1) / n
    hing = [(np.maximum(0, F - t) * okf).sum(1) / n for t in kn]
    pt = np.asarray(J['x'][r][..., 0], np.float32) * ok; z = pt / pt.sum(1, keepdims=True).clip(1e-6)
    wmean = (F * z[..., None]).sum(1)
    q = F[..., 7]; d0s = np.asarray(J['ext'][r][..., 3], np.float32) / np.maximum(np.asarray(J['ext'][r][..., 4], np.float32), 1e-6)
    ch = ok & (q != 0); dr = F[..., 4]
    i1 = np.argmax(np.where(ch, pt, -1), 1); i2 = np.argmax(np.where(ok & (z > .01), dr, -1), 1); i3 = np.argmax(np.where(ch, np.abs(d0s), -1), 1)
    sel = [np.take_along_axis(F[..., :NOWN + len(NBR)], i[:, None, None], 1)[:, 0] for i in (i1, i2, i3)]
    return np.concatenate([mean] + hing + [wmean] + sel + [np.log(n)], 1).astype(np.float32)


def names():
    base = PFEAT + NBR + PK; sel_base = PFEAT + NBR
    return ([f'mean[{b}]' for b in base] + [f'mean[max(0, {b} − t{k + 1})]' for k in range(5) for b in base] + [f'pT-weighted mean[{b}]' for b in base]
            + [f'{s}: {b}' for s in ('hardest charged', 'widest non-soft', 'most displaced charged') for b in sel_base] + ['ln n_particles'])


def build(n_fit=200000, chunk=4000, log=print):
    t0 = time.time(); model = ParTNetwork('full').model; model.eval(); DP.mkdir(parents=True, exist_ok=True); kn = None
    for which, n in (('fit', n_fit), ('dev', None)):
        J = jets('full', which); n = n or len(J['y']); out = None
        for a in range(0, n, chunk):
            r = np.arange(a, min(a + chunk, n)); F, ok = per_particle(J, r, model)
            if kn is None: kn = np.quantile(F[ok][::3], np.linspace(.15, .85, 5), axis=0).astype(np.float32)
            Q = pool(F, ok, J, r, kn)
            if out is None: out = np.lib.format.open_memmap(DP / f'{which}.npy', 'w+', np.float32, (n, Q.shape[1]))
            out[a:a + len(r)] = Q
            if a % 40000 == 0: log(f'  pooled {which}: {a + len(r)} of {n} jets, {Q.shape[1]} quantities, {time.time() - t0:.0f} s')
        out.flush()
    (DP / 'names.json').write_text(json.dumps(names())); np.save(DP / 'knots.npy', kn)
    log(f'pooled quantities: {len(names())} per jet, {time.time() - t0:.0f} s')


def check(n_fit=200000, epochs=40, device='mps', log=print):
    import torch
    from sklearn.preprocessing import QuantileTransformer
    from .ceiling_jet import matrix
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); keys = [k for k in Jf['Q']]
    Ef, Ed = matrix(Jf, keys)[:n_fit], matrix(Jd, keys); okc = np.isfinite(Ef).all(0) & (Ef.std(0) > 0); Ef, Ed = Ef[:, okc], Ed[:, okc]
    Pf, Pd = np.load(DP / 'fit.npy')[:n_fit], np.load(DP / 'dev.npy'); okp = np.isfinite(Pf).all(0) & (Pf.std(0) > 0); Pf, Pd = Pf[:, okp], Pd[:, okp]
    res = {}
    for name, (A, B) in dict(existing=(Ef, Ed), pooled=(Pf, Pd), both=(np.concatenate([Ef, Pf], 1), np.concatenate([Ed, Pd], 1))).items():
        t0 = time.time(); qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', subsample=100000, random_state=0).fit(A)
        T = lambda a: torch.from_numpy(np.clip(qt.transform(a), -5, 5).astype(np.float32)).to(device)
        xf, xd, pf = T(A), T(B), torch.from_numpy(np.asarray(Jf['P'][:n_fit], np.float32)).to(device)
        torch.manual_seed(0); net = torch.nn.Sequential(torch.nn.Linear(xf.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(),
                                                        torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, 10)).to(device)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (n_fit // 1024 + 1)); best = (0, 0)
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n_fit, device=device)
            for a in range(0, n_fit, 1024):
                i = perm[a:a + 1024]; loss = -(pf[i] * torch.log_softmax(net(xf[i]), 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            net.eval()
            with torch.no_grad(): pred = torch.cat([net(xd[a:a + 20000]).argmax(1) for a in range(0, len(xd), 20000)]).cpu().numpy()
            best = max(best, (float((pred == Jd['net']).mean()), float((pred == Jd['y']).mean())))
        res[name] = dict(n_quantities=int(A.shape[1]), agreement=best[0], accuracy=best[1])
        log(f'  ceiling {name} ({A.shape[1]} quantities): validation same class as ParT {100 * best[0]:.2f}%, accuracy {100 * best[1]:.2f}%, {time.time() - t0:.0f} s')
    return dict(experiment='S3', n_fit=n_fit, epochs=epochs, results=res)


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'build': build(*(int(v) for v in a))
    else:
        r = check(); OUT.mkdir(parents=True, exist_ok=True); (OUT / 'S3_pooled_ceiling.json').write_text(json.dumps(r, indent=1)); print('RESULT', json.dumps(r))
