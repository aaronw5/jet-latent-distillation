"""G4 — does ParT tell Hbb from Hcc by track displacement?
Prediction: among true Hbb and true Hcc jets that ParT calls Hbb or Hcc, P(ParT says Hbb) rises with the number of tracks
with |d0|/σ > 3 (n_sd0_above_3) in BOTH true classes, at fixed jet mass (10-GeV bins, 100–150 GeV) and jet-pT tercile.
Falsified if the stratified difference (≥ 4 displaced tracks vs ≤ 1) is ≤ 0 or < 3σ. Second variable: the 2nd-largest
signed impact parameter significance (sip_d0_2). Control: jet |η| terciles (no b/c physics expected). Also: how
different are b and c jets in these variables (truth)? Data: the 2M test jets."""
import numpy as np
from jetdistill.pipeline import jets
from jetdistill.config import CLASSES as CL
J = jets('full', 'full_test'); net = np.asarray(J['net']); y = np.asarray(J['y']); m = np.asarray(J['Q']['mass'], np.float32)
pt = np.asarray(J['Q']['sum_pt'], np.float32); eta = np.asarray(J['Q']['jet_abs_eta'], np.float32)
B, C = CL.index('Hbb'), CL.index('Hcc'); nd = np.asarray(J['Q']['n_sd0_above_3'], np.float32); s2 = np.asarray(J['Q']['sip_d0_2'], np.float32)
sel = ((y == B) | (y == C)) & ((net == B) | (net == C)) & (m > 100) & (m < 150)
print(f'jets: {sel.sum()} (true Hbb {(sel & (y == B)).sum()}, true Hcc {(sel & (y == C)).sum()})')
for nm, v in (('n tracks |d0|/σ>3', nd), ('2nd-largest signed d0 significance', s2)):
    print(f'truth: {nm}: median Hbb {np.median(v[sel & (y == B)]):.2f}, Hcc {np.median(v[sel & (y == C)]):.2f}')
pte = np.quantile(pt[sel], [0, 1 / 3, 2 / 3, 1])
def strat(var, lo_cut, hi_cut, label, c, extra=None):
    diffs, ws = [], []
    for a, b in zip(pte[:-1], pte[1:]):
        for mlo in range(100, 150, 10):
            base = sel & (y == c) & (pt >= a) & (pt <= b) & (m >= mlo) & (m < mlo + 10)
            lo_, hi_ = base & lo_cut, base & hi_cut
            if lo_.sum() < 200 or hi_.sum() < 200: continue
            rl, rh = (net[lo_] == B).mean(), (net[hi_] == B).mean(); v = rl * (1 - rl) / lo_.sum() + rh * (1 - rh) / hi_.sum()
            diffs.append(rh - rl); ws.append(1 / v)
    diffs, ws = np.array(diffs), np.array(ws); d = (diffs * ws).sum() / ws.sum(); e = 1 / np.sqrt(ws.sum())
    print(f'{label}, true {CL[c]}: {100 * d:+.2f} ± {100 * e:.2f} pt ({d / e:+.1f}σ, {len(diffs)} cells)')
for c in (B, C):
    k = sel & (y == c); print(f'true {CL[c]}: P(ParT says Hbb) by n displaced tracks: ' + ', '.join(f'{int(lo)}{"+" if hi > 50 else ""}: {100 * (net[k & (nd >= lo) & (nd < hi)] == B).mean():.1f}% (n={(k & (nd >= lo) & (nd < hi)).sum()})' for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 99))))
for c in (B, C):
    strat(nd, nd <= 1, nd >= 4, 'P(Hbb): ≥4 vs ≤1 displaced tracks, same mass bin and pT tercile', c)
    q = np.quantile(s2[sel], [1 / 3, 2 / 3]); strat(s2, s2 <= q[0], s2 >= q[1], 'P(Hbb): top vs bottom third of 2nd-largest d0 significance', c)
    q = np.quantile(eta[sel], [1 / 3, 2 / 3]); strat(eta, eta <= q[0], eta >= q[1], 'control, P(Hbb): top vs bottom third of jet |η|', c)
# true Hbb: too few jets with ≤ 1 displaced track for the stratified cells → ≤ 3 vs ≥ 6
strat(nd, nd <= 3, nd >= 6, 'P(Hbb): ≥6 vs ≤3 displaced tracks, same mass bin and pT tercile', B)
