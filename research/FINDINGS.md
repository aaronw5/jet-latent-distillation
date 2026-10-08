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
