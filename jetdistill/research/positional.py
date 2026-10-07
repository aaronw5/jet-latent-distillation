"""P1 — bypassing the attention: the 128 neurons as formulas of the particles in a FIXED order. Slots: the 12 hardest
particles (by pT) and the 6 most displaced charged tracks (by |d0|/σ, not among the 12), each slot's 38 per-particle
inputs as hinge terms (x, max(0, x − θ_k), max(0, θ_k − x)); plus the jet-level inputs of S11's class-token formula.
Least squares to the 128 neurons (ridge), ParT's last layer, then tuned toward ParT's probabilities. No attention
weights at all — the selection is the ordering.

  python -m jetdistill.research.positional [n_fit n_dev epochs]"""
import json, sys, time
import numpy as np
from ..pipeline import jets
from ..part.clsfit import particle_features
from ..part.network import ParTNetwork
from .nbr import nbr_features
from .heads import extract, rows_of_split, OUT
from .alpha_e2e import zfeat

NV, NPT, ND0 = 38, 12, 6


def slots(J, rows, knv):
    """(n, (NPT + ND0) * (11 NV + 1)) : every slot's 418 terms + a 'slot filled' flag"""
    F0, ok = particle_features(J, rows, ctx=[]); Fn = nbr_features(J['x'][rows], J['ext'][rows], J['jet'][rows]); X = np.concatenate([F0, Fn], -1)[..., :NV].astype(np.float32)
    d0s = np.abs(np.asarray(J['ext'][rows][..., 3], np.float32)) / np.maximum(np.asarray(J['ext'][rows][..., 4], np.float32), 1e-6); charged = F0[..., 7] != 0
    n = len(rows); out = np.zeros((n, NPT + ND0, 11 * NV + 1), np.float32)
    pt = np.where(ok, F0[..., 2], -np.inf); ordpt = np.argsort(-pt, 1)[:, :NPT]
    dsc = np.where(ok & charged, d0s, -np.inf); dsc[np.arange(n)[:, None], ordpt] = -np.inf; ordd = np.argsort(-dsc, 1)[:, :ND0]
    for s_, (idx, valid) in enumerate([(ordpt[:, k], np.take_along_axis(pt, ordpt[:, k:k + 1], 1)[:, 0] > -np.inf) for k in range(NPT)] + [(ordd[:, k], np.take_along_axis(dsc, ordd[:, k:k + 1], 1)[:, 0] > 0) for k in range(ND0)]):
        x = X[np.arange(n), idx] * valid[:, None]
        out[:, s_, :11 * NV] = np.concatenate([x] + [np.maximum(0, x - knv[k]) for k in range(5)] + [np.maximum(0, knv[k] - x) for k in range(5)], -1) * valid[:, None]; out[:, s_, -1] = valid
    return out.reshape(n, -1)


def run(n_fit=100000, n_dev=20000, epochs=200, lr=3e-4, lam=0.01, device='mps', log=print):
    import torch
    t0 = time.time(); model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    _, _, Mf, Lf, _ = extract(model, 'fit', n_fit, device); _, _, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rf, rd = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev); knv = np.load(OUT / 'S15p_model.npz')['knv']
    Zf, Zd = zfeat(Jf, Jd, n_fit, rd, Mf, Md); Bf = np.concatenate([slots(Jf, rf, knv), Zf], 1); Bd = np.concatenate([slots(Jd, rd, knv), Zd], 1); n, K = Bf.shape
    Hf, Hd = np.asarray(Jf['H'][rf], np.float32), np.asarray(Jd['H'][rd], np.float32); log(f'  {K} columns ({NPT} pT slots + {ND0} displaced slots × {11 * NV + 1}, + {Zf.shape[1]} jet-level), {time.time() - t0:.0f} s')
    T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device, dt); fc = model.fc
    G = Bf.T.astype(np.float64) @ Bf / n; d = np.sqrt(np.maximum(np.diag(G), 1e-12)); ok_ = d > 1e-9; Gs = G[np.ix_(ok_, ok_)] / d[ok_][:, None] / d[ok_][None]
    W0 = np.zeros((K, 128)); W0[ok_] = np.linalg.solve(Gs + 1e-3 * np.eye(ok_.sum()), (Bf[:, ok_].T.astype(np.float64) @ Hf / n) / d[ok_][:, None]) / d[ok_][:, None]
    ev, Q = np.linalg.eigh(Gs); ev = np.maximum(ev, 1e-3 * ev.max()); Mw = np.zeros((K, ok_.sum())); Mw[ok_] = Q / np.sqrt(ev)[None] / d[ok_][:, None]
    U = torch.nn.Parameter(T(np.sqrt(ev)[:, None] * (Q.T @ (d[ok_][:, None] * W0[ok_])))); Mt = T(Mw); Bft, Bdt, Hft = T(Bf), T(Bd), T(Hf); vh = Hft.var(0) + 1e-6; pf = torch.softmax(T(Lf), 1)
    def agree():
        with torch.no_grad(): return float((torch.cat([fc(Bdt[a:a + 5000] @ (Mt @ U)).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
    a0 = agree(); best = (a0, U.detach().clone(), 0); log(f'  least squares: {100 * a0:.2f}%, {time.time() - t0:.0f} s'); opt = torch.optim.Adam([U], lr)
    for ep in range(epochs):
        for a in np.random.default_rng(ep).permutation(np.arange(0, n, 5000)):
            y = Bft[a:a + 5000] @ (Mt @ U); loss = -(pf[a:a + 5000] * torch.log_softmax(fc(y), 1)).sum(1).mean() + lam * ((y - Hft[a:a + 5000]) ** 2 / vh).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9 or ep == epochs - 1:
            ag = agree()
            if ag > best[0]: best = (ag, U.detach().clone(), ep + 1)
            if ep % 50 == 49: log(f'  P1 epoch {ep + 1}: {100 * ag:.2f}% (best {100 * best[0]:.2f}%), {time.time() - t0:.0f} s')
    W = (Mt @ best[1]).cpu().numpy(); np.savez(OUT / 'P1_model.npz', W=W, knv=knv, npt=NPT, nd0=ND0)
    r = dict(experiment='P1', columns=int(K), least_squares=a0, best=best[0], best_epoch=best[2], seconds=time.time() - t0)
    (OUT / 'P1_positional.json').write_text(json.dumps(r, indent=1)); log(f'P1: the 128 neurons from the {NPT} hardest + {ND0} most displaced particles in fixed order, no attention: {100 * best[0]:.2f}% (least squares {100 * a0:.2f}%)'); return r


if __name__ == '__main__':
    run(*(int(v) for v in sys.argv[1:]))
