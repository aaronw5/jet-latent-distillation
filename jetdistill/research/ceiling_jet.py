"""E1 — how much of ParT can any function of the jet-level quantities reproduce? An MLP (GPU) trained toward ParT's
probabilities, and boosted trees (sklearn HistGradientBoosting, CPU) toward ParT's class, on the saved jet-level
quantities of the fitting jets; agreement with ParT on the validation jets.

  python -m jetdistill.research.ceiling_jet mlp|hgb [n_fit]"""
import json, sys, time, pathlib
import numpy as np
from ..pipeline import jets

OUT = pathlib.Path(__file__).resolve().parents[2] / 'research' / 'results'


def matrix(J, keys):
    return np.stack([np.asarray(J['Q'][k], np.float32) for k in keys], 1)


def data(n_fit):
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); keys = [k for k in Jf['Q']]
    Xf, Xd = matrix(Jf, keys)[:n_fit], matrix(Jd, keys)
    ok = np.isfinite(Xf).all(0) & (Xf.std(0) > 0); keys = [k for k, o in zip(keys, ok) if o]; Xf, Xd = Xf[:, ok], Xd[:, ok]
    return keys, Xf, Xd, Jf, Jd


def mlp(n_fit=300000, epochs=40, width=512, device='mps', log=print):
    import torch
    from sklearn.preprocessing import QuantileTransformer
    t0 = time.time(); keys, Xf, Xd, Jf, Jd = data(n_fit)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', subsample=200000, random_state=0).fit(Xf)
    T = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(device)
    xf, xd = T(np.clip(qt.transform(Xf), -5, 5)), T(np.clip(qt.transform(Xd), -5, 5)); pf = T(Jf['P'][:n_fit])
    net = torch.nn.Sequential(torch.nn.Linear(xf.shape[1], width), torch.nn.GELU(), torch.nn.Linear(width, width), torch.nn.GELU(),
                              torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, 10)).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (n_fit // 1024 + 1))
    best = (0, 0)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n_fit, device=device)
        for a in range(0, n_fit, 1024):
            i = perm[a:a + 1024]; loss = -(pf[i] * torch.log_softmax(net(xf[i]), 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad(): pred = torch.cat([net(xd[a:a + 20000]).argmax(1) for a in range(0, len(xd), 20000)]).cpu().numpy()
        ag = float((pred == Jd['net']).mean()); acc = float((pred == Jd['y']).mean()); best = max(best, (ag, acc))
        if ep % 5 == 4 or ep == epochs - 1: log(f'  MLP epoch {ep + 1}: validation same class as ParT {100 * ag:.2f}%, accuracy {100 * acc:.2f}%, {time.time() - t0:.0f} s')
    return dict(model='mlp', n_fit=n_fit, n_quantities=len(keys), width=width, epochs=epochs, agreement=best[0], accuracy=best[1], seconds=time.time() - t0)


def hgb(n_fit=300000, log=print):
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.time(); keys, Xf, Xd, Jf, Jd = data(n_fit)
    m = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=0)
    m.fit(Xf, Jf['net'][:n_fit]); pred = m.predict(Xd)
    r = dict(model='hgb', n_fit=n_fit, n_quantities=len(keys), iters=int(m.n_iter_), agreement=float((pred == Jd['net']).mean()), accuracy=float((pred == Jd['y']).mean()), seconds=time.time() - t0)
    log(f'  HGB ({m.n_iter_} iterations): validation same class as ParT {100 * r["agreement"]:.2f}%, accuracy {100 * r["accuracy"]:.2f}%, {r["seconds"]:.0f} s')
    return r


if __name__ == '__main__':
    kind = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 300000
    r = (mlp if kind == 'mlp' else hgb)(n); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'E1_{kind}_{n}.json').write_text(json.dumps(r, indent=1)); print('RESULT', json.dumps(r))
