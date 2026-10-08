# Discovery loop: what ParT and JEDI-linear learn

Each round: hypothesis → test (code below, data named) → numbers with uncertainties → verdict. Only verdicts marked
**holds** go on slides.

## G1 — ParT's W↔Z confusion is jet-mass resolution (holds)
- Hypothesis: the largest single ParT error (Zqq↔Wqq, 20 % of all its errors on the 2M test jets) comes from the overlap
  of the W and Z jet-mass peaks.
- Test: on the 2,000,000 test jets, the jet-mass resolution of true W / Z jets, and P(ParT calls a true Z “W”) and
  P(true W called “Z”) in jet-mass bins, binomial errors.
- Numbers: 68 % half-width of the jet mass 12.1 GeV (W), 12.8 GeV (Z) > the 10.8 GeV between m_Z and m_W.
  P(Z→W) 5.5 ± 0.2 % above 100 GeV, 18.7 ± 0.2 % at 85–90, 41.9 ± 0.6 % at 70–75 GeV; P(W→Z) 3.1 ± 0.1 % at 70–75,
  14.0 ± 0.2 % at 85–90, 26.8 ± 0.4 % at 95–100 GeV. Median mass of true Z called Z 92.7 GeV, of true Z called W 83.9 GeV
  (true W called W: 82.0 GeV).
- Formulas: where the class-token fit and ParT disagree on W/Z, the median mass is 87.2 / 87.8 GeV (the boundary),
  against 92.9 / 81.9 GeV where they agree.
- Verdict: holds. ParT decides W vs Z largely by jet mass; its errors are where the peaks overlap. Residual Z→W at high
  mass (5–8 %) needs another explanation → G2.

## G2 — inside the W/Z mass overlap, ParT uses the jet charge (holds)
- Hypothesis: for jets with 85 < m < 95 GeV, P(ParT says W) rises with |jet charge| (pT-weighted, κ = 0.5): a W → qq′ jet
  carries charge ±1, a Z jet none. Falsified if no rise once |η| and mass are held fixed.
- Data: the 2M test jets; 110,384 true W or Z jets in the window that ParT calls W or Z. Script: discovery/G2_jet_charge.py.
- First pass: P(W) rises with |Q| (true W 79.1 → 88.5 %, true Z 13.8 → 20.7 %), but the control (jet |η|) rises too
  (79.5 → 86.8 %, 14.8 → 20.3 %) → stratified.
- Stratified (same |η| tertile and 2-GeV mass bin; |Q| and |η| correlate at 0.018): highest vs lowest third of |Q|:
  **+6.83 ± 0.41 pt for true W (16.5σ), +4.02 ± 0.35 pt for true Z (11.4σ)**. |Q| alone separates W from Z here with
  AUC 0.561.
- Verdict: holds — ParT uses jet charge for W vs Z beyond mass. Separate, unexplained: at the same |Q| and mass, larger
  jet |η| also raises P(W) (+5.8 ± 0.4 / +2.9 ± 0.4 pt) → open observation O1.

## Audit of the first extras slides (rigour)
- The attention statistics on the extras (lepton weight, displacement profile, class-token share, effective particles)
  came from the page analyses on 20,000 validation jets; the footers said 100,000 — wrong. Recomputed on 100,000
  validation jets with errors (A1). The JEDI-linear extras quoted the formula's explanation (60,000 jets): removed until
  re-checked with ≥ 100,000 jets (G8). Slides that describe the fitted formulas themselves (type values, input rankings)
  are exact properties of the formula and are labelled so.
