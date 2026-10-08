"""S15 — the 128 class-token neurons predicted directly as per-particle formulas with ParT's attention weights (no
head structure, no downstream MLP): neuron_n = Σ_h Σ_i α_hi φ(x_i)·W_hn + α_h,cls c_hn + b_n over the 16 heads' pooled
terms (the same pooled basis as S5), then ParT's last layer (variant 'post': the targets are the neurons after the
final LayerNorm; variant 'pre': the class token before the final LN, ParT's LN applied). Least squares, then tuned
toward ParT's probabilities (+ λ·R on the neurons), whitened, 100k fitting jets, balanced 20k dev.

  python -m jetdistill.research.direct128 [post|pre] [n_fit n_dev epochs]"""
import json, os, sys, time
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from ..config import RESULTS
from .heads import extract, phi_pooled, rows_of_split, OUT


def run(variant='post', n_fit=100000, n_dev=20000, epochs=300, lr=3e-4, lam=0.01, device='mps', log=print):
    import torch
    t0 = time.time(); model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    Of, Af, Mf, Lf, _ = extract(model, 'fit', n_fit, device); Od, Ad, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rf, rd = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev)
    if os.environ.get('NOCLS'):                                                   # S16: no class-token share — the particles' weights renormalized to sum to 1, no α_cls terms
        for A_ in (Af, Ad): A_[..., 1:] /= np.maximum(1 - A_[..., :1], 1e-9); A_[..., 0] = 0
        log('  NOCLS: particle weights renormalized per head, the class token\'s share dropped')
    Pf, kn = phi_pooled(Jf, rf, Af); Pd, _ = phi_pooled(Jd, rd, Ad, kn); n, K = Pf.shape[0], Pf.shape[2]
    if os.environ.get('UNIFORM'):                                                  # control: no attention weights — every particle weighted equally, no class-token share
        for A_, M_ in ((Af, Mf), (Ad, Md)):
            A_[..., 0] = 0; A_[..., 1:] = (M_ / np.maximum(M_.sum(1, keepdims=True), 1))[None, :, None, :]
        log('  UNIFORM: every particle weighted 1/n in every head (no α)')
        Pf, kn = phi_pooled(Jf, rf, Af); Pd, _ = phi_pooled(Jd, rd, Ad, kn)
    hs = [int(h) for h in os.environ['HEADS'].split(',')] if os.environ.get('HEADS') else list(range(16))   # S19: only some heads' selections
    if len(hs) < 16: Pf, Pd = Pf[:, hs], Pd[:, hs]; log(f'  only heads {hs}')
    Bf, Bd = Pf.reshape(n, -1), Pd.reshape(n_dev, -1)                                               # all heads' pooled terms side by side
    # targets: the 128 neurons (after the final LN) or the class token before it
    D = RESULTS / '_cls' / 'full'; T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device, dt)
    Hf = np.asarray(Jf['H'][rf], np.float32); Hd = np.asarray(Jd['H'][rd], np.float32)
    if os.environ.get('LOGITS'):                                                   # S18: the 10 class logits directly, no last layer
        Yf, Yd = Lf.astype(np.float32), Ld.astype(np.float32); head = lambda y: y; variant = 'post'
    elif variant == 'pre':
        from .e4_layernorm import pre_ln
        Cf, _, _ = pre_ln('fit', n_fit, device); Cd_all, _, _ = pre_ln('dev', 100000, device); Yf, Yd = Cf, Cd_all[rd]; head = lambda y: model.fc(model.norm(y))
    else:
        Yf, Yd = Hf, Hd; head = lambda y: model.fc(y)
    with torch.no_grad(): chk = float((head(T(Yd)).argmax(1).cpu().numpy() == ref).mean())
    log(f'  {variant}: {Bf.shape[1]} pooled terms per jet; ParT\'s own targets through the last layer: {100 * chk:.2f}% (must be ≈ 100), {time.time() - t0:.0f} s')
    G = Bf.T.astype(np.float64) @ Bf / n; d = np.sqrt(np.maximum(np.diag(G), 1e-12)); ok_ = d > 1e-9
    NO = Yf.shape[1]; NB = Bf.shape[1]; W0 = np.zeros((NB, NO)); Gs = G[np.ix_(ok_, ok_)] / d[ok_][:, None] / d[ok_][None]
    W0[ok_] = np.linalg.solve(Gs + 1e-4 * np.eye(ok_.sum()), (Bf[:, ok_].T.astype(np.float64) @ Yf / n) / d[ok_][:, None]) / d[ok_][:, None]
    ev, Q = np.linalg.eigh(Gs); ev = np.maximum(ev, 1e-3 * ev.max()); Mw = np.zeros((NB, ok_.sum())); Mw[ok_] = Q / np.sqrt(ev)[None] / d[ok_][:, None]
    U = torch.nn.Parameter(T(np.sqrt(ev)[:, None] * (Q.T @ (d[ok_][:, None] * W0[ok_])))); Mt = T(Mw)
    Bft, Bdt = T(Bf), T(Bd); Yft = T(Yf); vy = Yft.var(0) + 1e-6; pf = torch.softmax(T(Lf), 1)
    def agree():
        with torch.no_grad(): return float((torch.cat([head(Bdt[a:a + 5000] @ (Mt @ U)).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
    a0 = agree(); best = (a0, U.detach().clone(), 0); log(f'  least squares: {100 * a0:.2f}%, {time.time() - t0:.0f} s'); opt = torch.optim.Adam([U], lr)
    for ep in range(epochs):
        for a in np.random.default_rng(ep).permutation(np.arange(0, n, 5000)):
            y = Bft[a:a + 5000] @ (Mt @ U); loss = -(pf[a:a + 5000] * torch.log_softmax(head(y), 1)).sum(1).mean() + lam * ((y - Yft[a:a + 5000]) ** 2 / vy).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9 or ep == epochs - 1:
            ag = agree()
            if ag > best[0]: best = (ag, U.detach().clone(), ep + 1)
            if ep % 50 == 49: log(f'  S15 {variant} epoch {ep + 1}: {100 * ag:.2f}% (best {100 * best[0]:.2f}%), {time.time() - t0:.0f} s')
    W = (Mt @ best[1]).cpu().numpy(); tag = os.environ.get('OUT_TAG', 'S15'); np.savez(OUT / f'{tag}_{variant}_model.npz', W=W, kn=kn, logits=bool(os.environ.get('LOGITS')))
    r = dict(experiment=f'{tag}_{variant}', heads=hs, terms=int(Bf.shape[1]), least_squares=a0, best=best[0], best_epoch=best[2], check=chk, seconds=time.time() - t0)
    (OUT / f'{tag}_{variant}.json').write_text(json.dumps(r, indent=1)); log(f'{tag} {variant}: the 128 neurons directly as per-particle formulas with ParT weights (no head structure / MLP): {100 * best[0]:.2f}% (least squares {100 * a0:.2f}%)'); return r


if __name__ == '__main__':
    a = sys.argv[1:]; run(a[0] if a else 'post', *(int(v) for v in a[1:]))
