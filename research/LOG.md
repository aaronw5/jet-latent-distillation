# Research log: reproducing ParT (ParT_full, JetClass)

Goal: a model that reproduces ParT's decisions (agreement = same class as ParT on its test/validation jets), as
interpretable as possible (formulas of physics quantities). Every experiment: hypothesis, setup, result, conclusion.
Numbers: validation = 100,000 dev jets; test = 2,000,000 test jets. Code: branch part-features, jetdistill/research/.
Results JSON: research/results/<id>.json.

## State at the start of the loop (2026-10-07 00:30)

| model | val agreement | test agreement | test acc | AUC |
|---|---|---|---|---|
| ParT | — | — | 0.8603 | 0.9877 |
| jet-level formula all_plus (1083 terms, MARS→tune→prune) | 80.09 | 80.09 | 0.7526 | 0.9665 |
| same, no-loss (1672 terms) | 80.21 | | 0.7538 | 0.9667 |
| all_plus_gelu (smooth terms, 869) | 80.19 | (step 4 running) | | |
| class fit (per-particle formula → ParT's frozen class blocks) | 78.90 | 78.70 | 0.7396 | 0.9640 |
| attention fit (own 2-layer class attention) | 75.48 | 75.39 | 0.7117 | 0.9534 |

Already tried, no gain: pair quantities in step 1 (R² 0.749 vs 0.756), λ = 0.1 / 1, tuning on true labels, GELU terms.
Known ceilings: ParT's class blocks fed with its own embeddings cut to the top 32 / 64 directions: 96.6 / 98.9 %.

## Experiments

### E1 — ceiling of the jet-level quantities (2026-10-07 00:40)
- Hypothesis: the formulas plateau at ~80% because either (a) the formula method loses information or (b) the jet-level
  quantities lack it. A flexible MLP on the same quantities separates the two (a measuring stick, not a candidate model).
- Setup: all saved jet-level quantities of results_part_plus (all_plus library), quantile-normalized; MLP 3×512 GELU,
  trained 40 epochs on 300k fitting jets toward ParT's probabilities; validation 100k. (A boosted-tree run was started and
  stopped: not informative beyond the MLP.)
- Result: **81.0 %** validation agreement (accuracy 75.9 %); formulas: 80.1–80.2 %.
- Conclusion: the formula method recovers ~99 % of what these quantities allow. The gap to ParT is **missing physics**
  in the quantities, not the method (as in JEDI-linear, where the method recovered the network once the right inputs were
  there). Next: find from ParT's weights what it computes that the quantities lack.

### B1 — the physics in ParT's pair-interaction weights (2026-10-07 01:05)
- ParT adds to every particle-attention logit a learned bias U_h = f_h(ln kT, ln z, ln ΔR, ln m²), one per head, shared
  by the 8 particle blocks. Measured on 1.3M real pairs (3000 jets): research/results/B1_pair_weights.json.
- Size: U_h's spread (1.3–3.6) is 2–3× the content part q·k/√d of block 1 (0.4–1.5) → **block-1 attention is mostly
  a learned physics kernel**. By block 8 the content part (1.4–2.0) is comparable.
- Simplicity: U_h is additive in the four pair variables to R² 0.85–0.99 (0.90–0.996 with pairwise products).
- Shapes (additive parts): heads 3, 5 favour collinear low-mass pairs (clustering); head 4 favours wide-angle pairs;
  head 8 intermediate ΔR ≈ 0.25 and harder kT; head 7 soft (low kT) high-mass pairs; heads 1, 2, 6 mixtures.
- Consequence: the particle context ParT builds can be written with ParT's own kernels (E5), instead of guessed
  neighbourhood features (E3).

### User's challenge (01:10): could be the method, not missing physics
- Correct reading of E1: no function of the *existing jet-level numbers* beats 81 %; methods that act before the
  summary into jet numbers (attention over particles, LayerNorm) are not ruled out. Tests: E4 (LayerNorm), E5 (attention
  with ParT's kernels), then ParT's extra class-attention operations (token LN, residual, MLP) in the attention fit.

### E2 — ceiling of the per-particle route with own inputs (2026-10-07 01:25)
- Per-particle MLP (38 own inputs + jet quantities) → ParT's frozen class blocks, 200k jets, 8 epochs: **81.8 %**
  (accuracy 76.7 %), above the jet-level ceiling (81.2 %): keeping ParT's class attention gains on its own.
- The pair-kernel reconstruction for E5 checked against B1 (per-head means/spreads agree within ~0.3).

## The plan (restated 01:30, after the user's question)
Goal: reproduce ParT — a formula model with ParT's structure making the same decisions (agreement → 100 %).
Method: replace ParT's components one at a time by formula versions, keep the rest of ParT, measure the agreement
drop (and the match of ParT's internal activations). Known: class blocks + last layer with ParT's embeddings (top 32
directions) 96.6 %; embeddings from each particle's own inputs ≤ 81.8 %. The loss is in the 8 particle-attention
blocks (context from other particles); block-1 attention is mostly the pair-physics kernel (B1).
Steps: (1) block 1 as formulas (attention = softmax(formula pair kernel + content), values/MLP as per-particle
formulas, ParT's LayerNorms as fixed operations), blocks 2–8 kept; (2) blocks 1–2, 1–3, … each checked; (3) all of ParT.
E3/E5 (context from hand-made neighbourhoods / from ParT's kernels) and E4 (final LayerNorm) are quick side checks.

## Plan corrected by the user (01:45): the JEDI-linear spirit
Not component-by-component reproduction. As for JEDI-linear: formulas of physics quantities fitted to the network's
last hidden neurons (MARS), tuned toward its probabilities, pruned — it recovered the network because the quantities
matched what the network computes (sums over particles). For ParT the neurons come from class attention: the
ParT-shaped quantities are attention-pooled sums Σᵢ αᵢ f(particleᵢ), with αᵢ selecting particles as ParT's heads do.
Steps: (1) ParT's class-attention weights per head → formulas of particle physics (its selection rule);
(2) pooled quantities with those weights (+ the class token's self-weight); (3) ceiling check (MLP on old + new
quantities); (4) JEDI pipeline. The component-swap code (replace.py) is kept but not pursued.

### E3 — per-particle route with neighbourhood physics (2026-10-07 02:00)
- As E2 + 20 neighbourhood features per particle (local density and pT share within ΔR 0.1/0.2/0.4, nearest
  neighbour ΔR / pT / kT, ΔR to the hardest, 2nd hardest, nearest displaced track, lepton, photon, its prong of the 3
  hardest directions with the prong's pT share, mass, charge, displaced count): **84.0 %** (accuracy 78.3 %) vs 81.8 %.
- Conclusion: physics context per particle carries real information ParT uses (+2.2 pt), and it is interpretable.

### S1 — ParT's class attention read as physics (2026-10-07 02:15)
research/results/S1_cls_attention.json (6k fitting jets for the score formulas, 4k check jets).
- Class block 1 (class token constant): selective heads, often with a large self-weight. Heads 1, 2, 3, 6: the ~2–4
  hardest charged particles; heads 4, 5: wide-angle particles (ΔR ≈ 0.4 vs 0.23); head 8: displaced tracks
  (|tanh d0| 0.22 vs 0.06; the b/c head); head 7: broad (~15 particles), self-weight ∝ multiplicity (corr −0.86).
  Self-weights: 0.40, 0.00, 0.58, 0.80, 0.00, 0.10, 0.35, 0.45.
- Class block 2: all 8 heads ≈ a plain average over all particles (≈37 effective), self-weight 0.02–0.03 falling
  with multiplicity (corr −0.6 to −0.8), i.e. 1/(N + c) normalization.
- Score formulas (within-jet R², own → +nbr → +pk): block 1 heads 0.29–0.83 → 0.42–0.87 → 0.49–0.90; block 2 low
  (0.11–0.42 → 0.23–0.62), but block-2 weights are nearly uniform, so their scores hardly matter.
- Conclusion (JEDI spirit): ParT's neurons ≈ a function of (mean over particles of a per-particle function of each
  embedding) + (a few selected particles: hardest charged, widest, most displaced). The quantities to give MARS:
  means over particles of per-particle physics functions (own + neighbourhood + ParT-kernel context, as hinges) and
  the features of the selected particles. Next: ceiling check of these quantities (step 3).

### E4 — the final LayerNorm (2026-10-07 02:30)
- Same 572 jet quantities, linear model tuned toward ParT: fitting the neurons after the final LN 75.88 % vs fitting the
  class token before it and applying ParT's LN 75.85 %. The hinge variant failed numerically (least-squares start
  broke, 9.64 %); not rerun: E1's MLP ceiling (81.2 %) already bounds any output-side operation on these quantities.
- Conclusion: the final LN is not the limit. LN matters inside the attention (on the particle embeddings); the class
  fit already uses ParT's own LNs there.

### S3 — quantities pooled like ParT's class attention (2026-10-07 02:50)
- 986 jet quantities: means over particles (and means above 5 thresholds, pT-weighted means) of 126 per-particle
  physics functions (own inputs, neighbourhood, ParT pair-kernel context) + the hardest charged, widest non-soft and
  most displaced particle's features. Flexible-fit check (MLP, 200k jets): existing 80.5 %, pooled 80.5 %, both 82.2 %.
- Conclusion: plain pooled jet quantities add only ~1.7 pt → not the way to close the gap by themselves.
- User: no more MLP checks (they were only a quick "is the information there" test); work with formulas directly.
  E5 (MLP with pair-kernel context) cancelled.

## S4 — per-head formulas (user's idea, 02:55)
Class-attention output = 2 blocks × 8 heads × 16 numbers; each head depends only on its own attention weights and the
values it pools; the rest is a fixed chain of ParT operations (out_proj, per-head scale, LayerNorms, MLP, residuals,
final LN, last layer). Targets: the 256 head outputs, fitted head by head (JEDI with per-head neurons), ParT's exact
downstream kept. Steps: oracle (ParT's true weights) → formula weights → head-by-head agreement cost → all, tuned.
