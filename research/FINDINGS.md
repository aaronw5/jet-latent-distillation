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

## A1–A3 — measurements behind the slides (100,000 validation jets, errors)
- A1 (ParT's attention): lepton weight b2h1 17.7 ± 0.1×, b2h8 7.8 ± 0.1×, b2h6 7.0 ± 0.1×; b2h2 gives tracks at 3–5σ
  8.2 ± 0.1× vs 0.68× prompt; class-token share b2h7 77 % (Tbl), 42 % (Hqql), ≤ 2 % other classes; Hgg jets get the most
  effective particles in 14 of 16 heads.
- A2 (class-token fit S15q): 18 active neurons; agreement 92.55 % on the 100k validation jets; neuron 98: W−Z +2.57 per s.d.
- A3 (attention-weights fit W1q): head b2h7's share formula reproduces the switch-off (Tbl 71 % vs ParT 77 %, Hqql 47 % vs
  42 %, correlation 0.72); largest terms all lepton quantities. Head b2h4 not reproduced (Hqql 18 % vs 46 %, corr 0.20);
  b2h8 partly (Wqq 21 % vs 67 %, corr 0.55).
- ParT neuron mass windows (100k): neuron 1 peak 122.5 GeV (half-max 108–138), neuron 103 172.5 (163–188), neuron 25 82.5
  (62–98, covers both W and Z), neuron 34 92.5 (88–98).

## G3 — the class-token fit's W/Z gap and |jet charge| (inconclusive)
- The fit's neurons read charge only through signed linear terms (1–2 % of a neuron; e.g. neuron 98, head b1h2: −1 → −0.57,
  +1 → +0.57), which cannot build |jet charge|. Prediction: its W/Z disagreement with ParT grows with |Q|.
- Stratified (as G2): ParT W / fit Z +0.56 ± 0.30 pt (1.9σ); ParT Z / fit W −0.60 ± 0.23 pt (2.7σ); control (attention-
  weights fit) −2.7 / +1.1 pt. Verdict: inconclusive; at most a small part of the gap.

## Corrections
- An earlier slide said neuron 17 "adds leptons": its formula gives muons −4.53 and electrons −2.74 (summed over heads) — wrong,
  removed. A slide said b2h2's formula scores "the most displaced track highest": its terms are mixed (one lowers the score
  near the jet maximum, another raises it by rank) — removed.
- No-α control (S20u): the class-token fit with every particle weighted 1/n: 69.78 % (least squares 60.47 %) vs 93.29 %
  with ParT's α — the attention weights carry 23.5 pt.
