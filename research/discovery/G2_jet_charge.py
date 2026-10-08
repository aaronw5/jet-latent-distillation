"""G2 — in the W/Z jet-mass overlap, does ParT use the jet charge?
Prediction: for true W and true Z jets with 85 < m < 95 GeV, P(ParT says Wqq | ParT says W or Z) rises with |jet charge|
(pT-weighted, κ = 0.5), for both true classes. Falsified if flat (within errors). Control: the same curve against a
charge-blind variable, the jet |η|, which should be flat.
Data: the 2,000,000 test jets; ParT's own predictions."""
import numpy as np
from jetdistill.pipeline import jets
from jetdistill.config import CLASSES as CL
J = jets('full', 'full_test'); net = np.asarray(J['net']); y = np.asarray(J['y']); m = np.asarray(J['Q']['mass'], np.float32)
q = np.abs(np.asarray(J['Q']['jet_charge_k05'], np.float32)); Z, W = CL.index('Zqq'), CL.index('Wqq')
win = (m > 85) & (m < 95) & ((y == W) | (y == Z)) & ((net == W) | (net == Z))
print(f'jets in the window: {win.sum()} (true W {(win & (y == W)).sum()}, true Z {(win & (y == Z)).sum()})')
edges = np.quantile(q[win], np.linspace(0, 1, 7))
for c, nm in ((W, 'true W'), (Z, 'true Z')):
    print(nm + ': P(ParT says W) by |jet charge| (κ=0.5) sextile')
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = win & (y == c) & (q >= lo) & (q <= hi); n = int(k.sum()); r = (net[k] == W).mean(); e = np.sqrt(r * (1 - r) / n)
        print(f'   |Q| {lo:.3f}–{hi:.3f}: {100 * r:5.1f} ± {100 * e:.1f}%  (n={n})')
# truth: how well does jet charge itself separate W from Z here (AUC)?
from sklearn.metrics import roc_auc_score
k = win; print(f'AUC of |jet charge| for true W vs true Z in the window: {roc_auc_score(y[k] == W, q[k]):.3f}')
ctrl = np.asarray(J['Q']['jet_abs_eta'], np.float32); ce = np.quantile(ctrl[win], np.linspace(0, 1, 7))
for c, nm in ((W, 'true W'), (Z, 'true Z')):
    print('control, ' + nm + ': P(ParT says W) by jet |η| sextile')
    for lo, hi in zip(ce[:-1], ce[1:]):
        k = win & (y == c) & (ctrl >= lo) & (ctrl <= hi); n = int(k.sum()); r = (net[k] == W).mean(); e = np.sqrt(r * (1 - r) / n)
        print(f'   |η| {lo:.2f}–{hi:.2f}: {100 * r:5.1f} ± {100 * e:.1f}%  (n={n})')

# --- stratified: the charge effect within |η| tertiles and 2-GeV mass bins (removes the |η| and mass confounders) ---
print('correlation of |jet charge| with jet |η| in the window:', round(float(np.corrcoef(q[win], ctrl[win])[0, 1]), 3))
te = np.quantile(ctrl[win], [0, 1 / 3, 2 / 3, 1]); qe = np.quantile(q[win], [0, 1 / 3, 2 / 3, 1])
for c, nm in ((W, 'true W'), (Z, 'true Z')):
    diffs, ws = [], []
    for a, b in zip(te[:-1], te[1:]):
        for mlo in range(85, 95, 2):
            base = win & (y == c) & (ctrl >= a) & (ctrl <= b) & (m >= mlo) & (m < mlo + 2)
            lo_, hi_ = base & (q <= qe[1]), base & (q >= qe[2])
            if lo_.sum() < 200 or hi_.sum() < 200: continue
            rl, rh = (net[lo_] == W).mean(), (net[hi_] == W).mean(); v = rl * (1 - rl) / lo_.sum() + rh * (1 - rh) / hi_.sum()
            diffs.append(rh - rl); ws.append(1 / v)
    diffs, ws = np.array(diffs), np.array(ws); d = (diffs * ws).sum() / ws.sum(); e = 1 / np.sqrt(ws.sum())
    print(f'{nm}: high − low third of |jet charge|, same |η| tertile and 2-GeV mass bin: {100 * d:+.2f} ± {100 * e:.2f} pt ({len(diffs)} cells, {d / e:.1f}σ)')
for c, nm in ((W, 'true W'), (Z, 'true Z')):
    diffs, ws = [], []
    for a, b in zip(qe[:-1], qe[1:]):
        for mlo in range(85, 95, 2):
            base = win & (y == c) & (q >= a) & (q <= b) & (m >= mlo) & (m < mlo + 2)
            lo_, hi_ = base & (ctrl <= te[1]), base & (ctrl >= te[2])
            if lo_.sum() < 200 or hi_.sum() < 200: continue
            rl, rh = (net[lo_] == W).mean(), (net[hi_] == W).mean(); v = rl * (1 - rl) / lo_.sum() + rh * (1 - rh) / hi_.sum()
            diffs.append(rh - rl); ws.append(1 / v)
    diffs, ws = np.array(diffs), np.array(ws); d = (diffs * ws).sum() / ws.sum(); e = 1 / np.sqrt(ws.sum())
    print(f'{nm}: high − low third of jet |η|, same charge tertile and mass bin: {100 * d:+.2f} ± {100 * e:.2f} pt ({d / e:.1f}σ)')
