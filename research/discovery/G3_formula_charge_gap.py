"""G3 — the class-token fit's W/Z gap and the jet charge.
The fit's neurons read the particle charge only through signed, linear terms (1–2 % of a neuron), so they cannot build
|jet charge|, which ParT uses (G2). Prediction: among jets ParT calls W in the overlap window (85–95 GeV), the fit
disagrees (says Z) more often at large |jet charge|... no: ParT's extra W calls at large |Q| are the ones the fit cannot
follow → P(fit says Z | ParT says W) rises with |Q|; and P(fit says W | ParT says Z) falls with |Q|. Stratified by |η|
third and 2-GeV mass bin. Control: the attention-weights fit (ParT's own values), which can follow ParT's charge use →
expected flatter. Data: the 2M test jets."""
import numpy as np
from jetdistill.pipeline import jets
from jetdistill.config import CLASSES as CL
J = jets('full', 'full_test'); net = np.asarray(J['net']); m = np.asarray(J['Q']['mass'], np.float32)
q = np.abs(np.asarray(J['Q']['jet_charge_k05'], np.float32)); eta = np.asarray(J['Q']['jet_abs_eta'], np.float32); Z, W = CL.index('Zqq'), CL.index('Wqq')
win = (m > 85) & (m < 95); te = np.quantile(eta[win], [0, 1 / 3, 2 / 3, 1]); qe = np.quantile(q[win], [0, 1 / 3, 2 / 3, 1])
for tag in ('S15q', 'W1p'):
    f = np.load(f'research/results/{tag}_test_logits.npy').astype(np.float32).argmax(1)
    for pc, oc, nm in ((W, Z, 'ParT says W, fit says Z'), (Z, W, 'ParT says Z, fit says W')):
        diffs, ws, rl_all, rh_all = [], [], [], []
        for a, b in zip(te[:-1], te[1:]):
            for mlo in range(85, 95, 2):
                base = win & (net == pc) & (eta >= a) & (eta <= b) & (m >= mlo) & (m < mlo + 2)
                lo_, hi_ = base & (q <= qe[1]), base & (q >= qe[2])
                if lo_.sum() < 200 or hi_.sum() < 200: continue
                rl, rh = (f[lo_] == oc).mean(), (f[hi_] == oc).mean(); v = rl * (1 - rl) / lo_.sum() + rh * (1 - rh) / hi_.sum()
                diffs.append(rh - rl); ws.append(1 / v); rl_all.append(rl); rh_all.append(rh)
        diffs, ws = np.array(diffs), np.array(ws); d = (diffs * ws).sum() / ws.sum(); e = 1 / np.sqrt(ws.sum())
        print(f'{tag}: {nm}: low |Q| third {100 * np.mean(rl_all):.1f}%, high third {100 * np.mean(rh_all):.1f}%; stratified difference {100 * d:+.2f} ± {100 * e:.2f} pt ({d / e:+.1f}σ); jets {int((win & (net == pc)).sum())}')
