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
- Correction (03:05): in ParT's class attention the query is the raw class token (attn(x_cls, u, u); only keys and
  values are LayerNorm-ed). S1 had normalized the query → S1's weights were somewhat off; S1 rerun as S1b. The exact
  downstream reimplementation (heads.py) reproduces ParT: 100 % same class, logits within 0.002.

### S1b — class attention read as physics, corrected (2026-10-07 03:15) — supersedes S1
research/results/S1_cls_attention.json (overwritten by the corrected run).
- Block 1: all heads broad (22–29 effective particles of ~39), mild preference for harder / charged particles (head 8
  slightly for displaced); self-weights 0.00–0.15, falling with multiplicity (corr −0.5 to −0.8). Score formulas
  (within-jet R², own → +nbr → +pk): 0.26–0.83 → 0.41–0.87 → 0.49–0.91.
- Block 2: more selective (8–17 effective particles), harder and charged; self-weights 0.01–0.25; scores poorly given
  by per-particle physics (0.06–0.37 → 0.10–0.45 → 0.13–0.57): the block-2 query is block 1's output, so which
  particles matter depends on the whole jet (bilinear: Σ_k g_k(jet) f_k(particle)).
- The S1 statements "hardest charged / wide-angle / displaced heads" and "block 2 = plain average" are withdrawn
  (artefacts of the normalized query).

### S4b — per-head formulas, ParT's own attention weights (2026-10-07 03:20)
research/results/S4_heads_oracle.json. 40k fitting / 20k validation jets; per head: 230 pooled terms (Σ_i α_hi ·
hinge(own 18 + neighbourhood 20 features, 5 thresholds), self-weight, 1); least squares only.
- One head as a formula, the rest ParT: 97.1–99.8 %. All 8 block-2 heads: 98.4 %; all 8 block-1 heads: 86.1 %;
  **all 16 heads: 82.5 %** — above every previous formula result (80.2 %), before any tuning.
- Head-output R² is low (medians 0.03–0.58), yet agreement is high: the downstream (LNs) is forgiving, and least
  squares spends effort equally on all 256 outputs → tuning toward ParT's probabilities next (S5).
- Plan: S5 tune (JEDI step 2) through ParT's exact downstream; S6 formula weights: score = query·key, block 1's query
  is a constant (key per particle → formula), block 2's query = ParT's own query projection of the formula class token.

### S5 — per-head formulas tuned toward ParT (2026-10-07 03:35)
- First run (plain coefficients scaled by term spread, Adam lr 1e-3) diverged (82.5 → 0.8 %): correlated hinge terms.
- S5b: per head, tuned in whitened terms (Gram eigenbasis), lr 3e-4, λ = 0.01 on the head outputs, 40k fitting /
  20k validation jets: 82.5 % (least squares) → 84.7 / 86.1 / 87.2 / 88.1 / 88.8 / 89.5 / 89.8 / 90.1 / 90.4 /
  **90.6 %** after 100 epochs, still rising.
- With ParT's own attention weights, the head values are sums over particles of per-particle physics formulas
  (JEDI-linear's structure) and reproduce ParT to > 90 %. The open part: the weights as formulas (S6).
- S5c: 100k fitting jets, 500 epochs: least squares 82.7 % → 86.9 (10) / 91.5 (50) / 92.6 (100) / 92.9 (200) /
  93.0 (300) / **93.2 % (400)** / 93.2 (500). The ceiling of per-head formula values with ParT's own weights ≈ 93 %.

### S6 — formula keys and values, ParT's own queries (2026-10-07 04:05)
- Per particle, hinge terms (757) of own + neighbourhood + ParT pair-kernel context → its keys and values in both class
  blocks (512); ParT's queries, self key/value, softmax and downstream (check with ParT's own keys/values: 100 %).
  Key/value R² (medians): block 1 0.47 / 0.56, block 2 0.39 / 0.41. Least squares 65.9 %; tuned (Adam, plain
  coefficients, λ = 0.1 on keys/values) oscillating 73.5–76.1 %, best **76.1 %**.

### S7 — how precise must the class-attention weights be? (04:10)
- ParT's own values, only the weights swapped: block 2 uniform 99.1 %; block 1 uniform 92.8 %; block 1 particles
  uniform with ParT's self-weight 88.7 %. The information is in the values; the weights matter little (block 2 hardly
  at all). Reading of S6: wrong-but-peaked formula weights hurt far more than uniform ones.
- Next S8: S5 (per-head formula values) with uniform weights — means over particles of per-particle physics functions,
  ParT's exact downstream: JEDI-linear's structure.
- S7 (continued): ParT's values with weights from per-particle score formulas (hinge terms of own + neighbourhood +
  pair-kernel context, within-jet least squares, ParT's per-jet mean score kept): block 1 **95.8 %**, block 2 99.4 %.

### S8 — per-head formula values with uniform weights (2026-10-07 04:40)
- Both blocks uniform: least squares 62.1 % → tuned **67.8 %** (300 epochs, 100k jets). Block 2 uniform, block 1
  ParT's weights: **91.2 %**.
- Reading: with ParT's values the weights hardly matter (S7), with formula values they matter a lot — ParT's values
  carry each particle's context, formula values only its own physics, so the selection must do the work. Block 2 can
  stay a plain average (−2 pt); block 1's weights must be formulas (S9).

### Correction (2026-10-07 05:20): the validation jets of S4–S9 were not class-balanced
- The extracted dev jets are stored in the sorted order of the dev split (by file); S4–S9 took the first 20k (S7:
  10k, S6: 20k): 9472 H→bb, 8411 H→cc, 154 QCD, … of 20k. Those agreements (93.2 %, 95.8 %, 91.2 %, 76.1 %,
  67.8 %, 75.8 %) are therefore not comparable to the 80 % of the jet-level formulas (all 100k dev jets) and are
  withdrawn. The fit jets (first 100k of the shuffled fit split) are balanced. Fix: a random balanced 20k of the 100k
  dev jets (heads.rows_of_split, seed 0) everywhere; S5c, S7, S8b, S9/S10 rerun (suffix -bal).
- S9 (skewed dev, for the record): least squares 67.9 % → tuned 75.8 % (score formulas R² 0.51–0.90).
- GELU run finished and published (all_plus_gelu page).

### all_plus_gelu — test results (clock 01:49)
- 2M test jets: 869 terms 80.20 % agreement, accuracy 0.7536, AUC 0.9666; no-loss 1649 terms 80.34 % / 0.7546 /
  0.9669; step-4 688 terms 80.09 % / 0.7525 / 0.9663. Hinges (all_plus): 80.09 % / 0.7526 with 1083 terms.
  Same quality, ~20 % fewer terms. Page published.
- Note: the wall-clock times written in this log before this entry were estimates and run a few hours ahead of the
  machine's clock (logs/timing.log has the real times).

### Balanced reruns (clock ~02:10)
- S5c-bal (per-head formula values, ParT's weights, 100k fit, balanced 20k dev): least squares 83.7 % → **93.2 %**
  (300 epochs). The > 90 % result stands.
- S7-bal (ParT's values, weights swapped): block 1 uniform **78.4 %** (skewed dev had said 92.8 %); block 1 particles
  uniform with ParT's self-weight 88.9 %; block 2 uniform 96.4 % (was 99.1 %). The weights matter more than the
  skewed sample suggested, block 1's self-weight in particular.
- S7-bal (cont.): ParT's values with weights from per-particle score formulas (within-jet fit, ParT's per-jet mean
  score kept): block 1 **96.0 %**, block 2 98.2 %. The relative weighting is formula-shaped; the per-jet level of the
  scores (the self-weight balance; in block 1 the self score is a constant, so the level is the formula's job) is what
  S9 lacked (its self score was fixed at 0 with a least-squares intercept) — S10 tunes it.
- S8b-bal (formula values, ParT's block-1 weights, block 2 uniform): 80.6 % → **91.6 %** (300 epochs). A plain
  average in block 2 costs ~1.6 pt against ParT's block-2 weights (93.2 %).

### S10 — all formulas, tuned jointly (clock 02:10–02:25)
- Block-1 weights from per-particle score formulas (757 terms: own + neighbourhood + pair-kernel context; self score
  0), block 2 a plain average, per-head values from own + neighbourhood terms (228); joint Adam (whitened), 100k fit,
  balanced 20k dev: least squares 69.4 % → 81.1 (10) / 82.2 (30) / 82.6 (50) / **83.0 % (100)**, still creeping.
- The first stand-alone formula model above the jet-level formulas (80.2 %). Against S8b (91.6 % with ParT's block-1
  weights) the formula weights cost ~8.6 pt. S10b: values on all 126 features, 150k jets.

## Pages for the per-head models (requirements, from the user)
Each model gets its own page reflecting the per-head structure: an explorer by head (block × head) showing the
selection (score formula / weight profile; ParT's where kept, labelled), the 16 value neurons as formulas with terms by
importance and response curves, the head's importance (ablation) and classes; a card stating exactly what is kept of
ParT; the paper metrics vs ParT and the jet-level formulas; per class; pruning path; files. The old 128-neuron
explorer does not apply to these models.

### Readability of the all-formula model (clock ~02:50)
- S10b (values on all 126 features, 150k jets): least squares 75.5 % → 82.1 (5) / 82.8 (10) / **83.1 % (30)**, then
  flat/slightly down (82.6 % at 55). Context in the values helps the start (+6 pt) but the ceiling stays ≈ 83 %.
- A worked example (block-1 head 3 of S10 on a t→bqq jet) reads well: the muon with |d0|/σ = 3.1 gets weight 0.145,
  the leading photon 0.119, the class token 0.117, soft hadrons 0.01–0.02. But the score formula itself is not
  readable: its ten largest additive pieces (the kernel-context type fractions, nearly collinear) each span ±20 and
  cancel to a total spread of 0.7 — unpenalized least squares + whitened tuning on near-duplicate inputs.
- Fixes (S10c): degenerate hinge columns masked (duplicate thresholds, thresholds at a clip value), ridge 1e-3 in the
  starts, L1 = 0.01 on the raw coefficients during tuning, whitening eigenvalue floor 1e-3; then pruning by input.

### Pages for the per-head models (clock ~02:30–02:50)
- heads_S5 and heads_S8_uniform2 published (validation numbers; test metrics added when the 2M pass ends). A `%%`
  left in the page's JavaScript broke every button in the first version — fixed (node --check on the script is now part
  of the build check). Added on the user's request: a card showing in code how the class scores are composed; an exact
  Python formulation per model ({tag}_formulas.py + the .npz) and a Python-like text of every value neuron (terms
  covering 99 % of its weight); math definitions of all per-particle inputs (e.g. “ΔR nearest displaced” = min ΔR_ij
  over the other particles j with q_j ≠ 0 and |d0_j|/σ(d0_j) > 3, 1.5 if none).
- Compute rule from the user: only lines that can reach > 90 % agreement run (S5, S8b and their refinements);
  the all-formula line (≈ 83 %) is parked.

### S5 and S8b on the 2M test jets (clock ~03:25)
- S5 (per-head formula values, ParT's attention weights): **93.16 %** same class as ParT, accuracy 0.8427, AUC 0.9845
  (ParT 0.8603 / 0.9877). S8b (block 2 a plain average): 91.50 %, 0.8344, 0.9831. Validation and test agree.

### S11 — α as formulas on top of S5 (the user's idea: α is a selection heuristic; clock ~03:15)
- α_i = (1 − α_cls)·softmax_i(g(x_i)) per head: g a per-particle score formula (own + neighbourhood + pair-kernel
  context + within-jet relative inputs: rank, x − jet max, standardized), least squares to ParT's log(α_i/α_cls)
  centred within each jet and weighted by ParT's α; α_cls a jet-level ridge formula of the jet quantities (+ ln n).
- Smoke test (2k fitting jets): with ParT's values — formula ranking + ParT α_cls (block 1) 97.0 %, + formula α_cls
  96.8 %, both blocks all formula 95.95 %. With S5's value formulas — ParT's α 93.9 %, formula ranking + ParT α_cls
  91.9 %, block 1 all formula 82.9 %: S5's values lean on the class-token term c·α_cls (large c), so α_cls errors are
  amplified → S12: re-tune S5's values on the formula α (stagewise).
- S11 full (score formulas on 20k jets, α_cls on 100k, balanced 20k dev): per-head weights off ParT's by 6–21 %
  (total variation) in block 1, 39–61 % in block 2; α_cls logit R² 0.76–0.98 (block 1), 0.32–0.84 (block 2).
  Agreement — ParT's values: formula ranking + ParT α_cls (block 1) 96.89 %, + formula α_cls 96.67 %, both blocks
  all formula **95.74 %**. S5's values: ParT's α 93.16 %, block-1 formula α 83.91 %, both 74.84 %.
- Reading: α as formulas is good enough when the values are right (95.7 % with ParT's values, everything else
  formula); S5's values were fit to ParT's α and do not transfer → S12 re-tunes them on the formula α.

### S12 — S5's values re-tuned on the S11 formula α (stagewise all-formula; clock ~03:45)
- Least squares 75.5 % → best **82.6 %** (epoch 90), flat after: the same wall as S10 (83.0 %).
- The 2 × 2 picture (balanced dev): ParT values + ParT α 100 %; ParT values + formula α 95.7 %; formula values +
  ParT α 93.2 %; formula values + formula α ≈ 83 %. The losses compound rather than add (additive would give ≈ 89 %):
  ParT's α brings in context from the particle embeddings (each particle's relation to the whole jet); once α is also a
  formula of the same per-particle inputs, no context enters, and the model falls back to the information limit of
  those inputs — a flexible per-particle model with them reached 84 % (E3).
- Consequence (90 % rule): the all-formula line needs better per-particle context inputs, not a better fit.
  Candidates: multi-hop pair-kernel context (kernel averages of kernel averages, mimicking stacked attention),
  kernel-weighted sums of more properties at several scales, the particle's prong / neighbourhood composition.
  Cheap formula-based test: do they bring S11's weights closer to ParT's (total variation) and raise formula values +
  ParT α above 93.2 %?

### S5rp — the pruned readable S5 on the 2M test jets (clock ~03:50)
- **93.30 %** same class as ParT, accuracy **0.8435**, AUC 0.9847 (ParT 0.8603 / 0.9877) — with 380 (head, input)
  pairs (17–29 inputs per head) instead of 608: smaller and slightly better than S5 (93.16 %). The headline model.

### Cancellation in the value neurons (user's question, clock ~04:00)
- Measured on the balanced dev jets, median over (head, neuron): inside one input, net / Σ|terms| = 0.08 (S5) →
  0.20 (S5rp); between inputs, neuron / Σ|inputs| = 0.07 → 0.13. The ridge, L1, duplicate removal and pruning helped
  but most magnitude still cancels. Inside an input this is largely the hinge parametrization (a hinge coefficient
  is a change of slope) — fixed by displaying each input as slopes per segment. Between inputs it is real:
  correlated inputs offset each other.
- S13: S5rp re-tuned with an anti-cancellation penalty Σ_f sd(C_f) − sd(Σ_f C_f) (relative), keeping the least
  cancelling model within 0.2 pt of S5rp.

### S13 — anti-cancellation re-tune of S5rp (clock ~03:35)
- 93.27 % (balanced dev), cancellation ratios inside 0.08 / between 0.68 (S5rp: 0.20 / 0.13): the inputs of a neuron
  now mostly reinforce each other. ≥ 93 %: 2M-jet test and page (S5rpc) queued.

## To do next (user, 03:40)
- Write each value neuron's formula as if-statements (the piecewise-linear pieces are exactly that: one branch per
  interval between kinks), as in JEDI-linear.
- A per-neuron explorer in the style of the JEDI-linear page (per head → per value neuron): drop-downs and the useful
  views of that page (the formula as if-statements, the inputs by importance, response curves, which jets / classes
  move it, worked examples), adapted to the per-head structure (weights α, sum over particles).
- Per neuron, a qualitative description of what it looks at (as the JEDI-linear explorer's explanations): in words,
  which particles (the head's selection) and which of their properties raise / lower it, and which classes it serves.
- Combine like terms (user): merge adjacent segments of an input's piece whose slopes barely differ (e.g. ΔR with
  slopes 24.4 / 26.8 / 22.4 / 27.3 / 20.9 / 19.8 → one slope), merge repeated thresholds in the stored model, combine
  inputs that carry the same information where the physics allows; re-tune; keep only within 0.1 pt (as JEDI's
  simplification step).

### S14 — at most one term per input per value neuron (user's goal; clock ~05:00)
- From S5rpc: per (head, neuron, input) the single best term (x, max(0, x − θ) or max(0, θ − x) at one of 5
  thresholds; the "below" hinges pooled exactly from the existing sums), least squares 89.28 %, re-tuned **93.06 %**
  (balanced dev) with 6080 terms instead of 36480 nonzero coefficients. (Two re-tuning runs blew up: the constant
  bias column's zero spread made its coefficient scale 1e6 — fixed.) Formulas shown as symbolic math, one term per
  input. 2M-jet test and page queued.
- S14 (S5rpc1) on the 2M test jets: **93.01 %** same class as ParT, accuracy 0.8424, AUC 0.9845 — at most one term per
  input per neuron (6080 terms) costs 0.2 pt against S5rpc (93.21 %). Page with the per-neuron explorer (JEDI-style
  facts, if-statements, symbolic formula) publishing.

### JEDI-linear: at most one term per observable per neuron (repo ~/Documents/jet-latent-distillation, branch one-term)
- n8, test file: 805 terms 90.57 % → 589 terms 89.75 % (accuracy 65.45 %); the smaller 399 → 328 terms, 90.06 % →
  89.28 %. n64: data found in ~/Documents/jedi-distill/data (splits identical to the existing n64 caches); step 1 +
  tuning running, then the one-term step.
- JEDI-linear n64 (step 1 12 min, tuning 13 min; main formula 641 terms, 94.14 % on the test file — the published run
  had 690 at 94.18 %): one term per observable → **427 terms, 93.30 %** (accuracy 81.03 %).

## Research loop A — the best fit for the attention weights α (user: "find the best fit"; clock ~08:00)
All-formula end-to-end is stuck at ~83 % (S10, S12) while formula α with ParT's values reaches 95.7 % (S11) and formula
values with ParT's α 93 %: the losses compound — ParT's α carries each particle's context, our inputs one hop of it
(E3: a flexible per-particle model on them through ParT's class attention reaches 84 %). Steps (each measured as
(1) total-variation distance of the weights from ParT's, (2) formula α + ParT's values, (3) formula α + formula values
re-tuned, end to end):
- A1 richer context: 2nd hop of ParT's pair kernels and pT-weighted kernels (302 inputs per particle).
- A2 decision-focused α: score formulas (and α_cls) tuned end to end through ParT's downstream with ParT's values fixed.
- A3 learnable pair-interaction formula: s_i = g(x_i) + Σ_j K(ΔR_ij, kT_ij, z_ij, m²_ij)·h(x_j), K, g, h small formulas.
- A4 jet-conditioned scores: Σ_k a_k(jet)·b_k(particle) (products with a few jet quantities).
- A5 selection by rank under a formula score.
- A1 (2nd-hop + pT-weighted kernel context, 302 inputs; score formulas on 10k jets): block-1 weights 5.3–18.6 % off
  ParT's (S11: 5.7–19.5 %). ParT values + formula α: block 1 97.14 % (96.89), both blocks **95.95 %** (95.74);
  S5 values + formula α both 75.77 % (74.84). A small gain: ParT's own kernels applied twice do not capture what its
  8 blocks build. End-to-end re-tune (A1_values) next, then A2 (decision-focused) and A3 (learnable pair interaction).
- A2 (decision-focused weight formulas, ParT's values fixed, 20k jets, 30 epochs): from S11 95.73 → **96.84 %**; from
  A1 (richer context) 95.95 → **97.04 %**. (Writing A2h's weights for 100k jets ran out of GPU memory → fixed with
  batches, rewritten from the saved formulas; a run that started without a weight source was stopped.)
- A2h end to end (formula values re-tuned on A2h's weights): 74.4 % → best **82.5 %** (epoch 130). Same wall as S10 /
  S12: weights that serve ParT's values at 97 % do not lift formula values — plain per-particle values only work with
  ParT's own weights, which carry the context. Next: A3 (learned pair interaction in the weights), then A6 (the same
  learned context in the values).
- A3 (learnable pair interaction in the weights, ParT values fixed): no gain over A2h — 97.04 → 96.4–96.8 % while
  training; best = the start. With ParT's values the weights are not the limit. Its end-to-end was skipped (identical
  weights to A2h).
- A6 (learned pair context in weights AND values, all formulas, 40k jets): 82.5 % → best **83.2 %** (epoch 15), 82.7 %
  at 40. One learned hop ≈ the hand-made neighbourhood (E3 84 %): the same information limit.
- A7 queued: L = 3 stacked learned hops (formula message passing: kernel formula per hop + linear maps), in weights
  and values.
- Rule applied (user): no more runs that cannot reach 90 %. A1's end-to-end cancelled (≤ A2h). A4 (jet-conditioned
  scores) and A5 (rank selection) dropped: they change the weights, and the weights are not the limit — formula
  weights already serve ParT's values at 97 % (A2h); every all-formula variant with one hop of context stops at
  83 %. The only remaining candidate is multi-hop context (A7); if it fails too, the stand-alone-formula line ends
  here and the > 90 % models are the S5 family (ParT's selection kept): S5rpc1 93.0 % on the 2M test jets.

### Why 83 % when ParT's values give 97 % — the shuffle test (clock ~10:55)
Same formula values (S5rpc1), same downstream, 20k balanced dev jets; only the weights change: ParT's own 93.06 %;
ParT's own **shuffled among the particles of each jet** (same distribution, wrong particles) **60.96 %**; ParT's
class-token share with uniform particles 63.21 %; A2h formula weights with these values untouched 19.42 % (82.5 %
after re-tuning the values on them). → 32 pt of the 93 % come from ParT's weights picking the right particles per jet
(context from its embeddings), not from their shape. With ParT's values the context is in the values and any smooth
weighting works (the 97 % was the easy test); with per-particle formula values the weights must carry the context,
and formula weights (per-particle too) cannot.
- A7 (3 learned hops): tracking A6 (≈ 83 %) — learned context from scratch does not recover the selection either.
- A7 (3 stacked learned hops, 40 epochs): best 83.16 % — no better than one hop.

### Is the class-token selection a cut? (user's question; clock ~11:00)
Formula values (S5rpc1), same downstream, only the weights changed: ParT's own 93.06 %; **ordering only** (ParT's rank
of each particle → the average weight at that rank) **90.39 %**; weights flattened (√) 89.49 %, sharpened (²) 87.28 %;
hard cut uniform over ParT's top 10 / 5 / 3 per head 79.19 / 76.68 / 73.41 %. → Not a hard cut but a graded
ranking: the per-jet ORDER of the particles carries 90 of the 93 points, the magnitudes < 3. New target (A9): a
per-head formula RANKER (pairwise ranking loss against ParT's order within each jet) + a fixed per-rank weight
profile — a far easier formula target than the softmax scores.

## Direction (user, ~11:10): α is a per-particle classifier (each particle's weight from its own features); the 83 %
models are starting points — train and tune the thresholds until > 90 %. Identify the most promising and tune hard.
- A9: formula ranker (ordering carries 90 of the 93 points) → then S5 values re-tuned on it.
- A10: the all-formula per-head model with coefficients AND thresholds learnable, long cosine schedule, 100k jets,
  started from the best weights (A2h / A9) and values (S12A2h).
- A8 probe (keys and values of ParT's class attention from k-hop physics, ridge on hinge terms; agreement with the
  predicted keys+values in the class attention): own 60.9 %, +1 kernel hop 71.6 %, +2nd hop/pT-weighted 74.3 %,
  +3 plain hops 75.1 %; median R² of block-1 keys/values 0.37/0.44 → 0.51/0.61. Context helps but saturates.
- A9 (formula ranker, pairwise loss vs ParT's order, 30k jets, 20 epochs): Kendall τ block 1 0.48–0.81, block 2
  0.08–0.45. Rank-profile α: ParT values 95.4 %; S5rpc1 formula values block 1 82.6 %, both 69.9 % — no better than
  the score formulas: the ordering is as hard to get from per-particle inputs as the weights.
- A10 (coefficients + thresholds learnable, from A2h / S12A2h, 100k jets, 60 epochs): start 82.51 %, running.
- A10 (A2h start, coefficients + thresholds learnable, 100k jets, 60 epochs, one-cycle): 82.51 → **84.05 %** (epoch 32);
  thresholds moved 0.15 σ (scores), 0.02 (values). The best all-formula number, still ≈ 11 pt under the ParT-α models.
- A9 end to end (ranker α, values re-tuned): 82.15 %.

### S15 — the 128 class-token neurons predicted directly (user's question; clock ~11:25)
- Each of the 128 neurons = Σ over the 16 heads' pooled per-particle terms (ParT's α) · coefficients + α_cls terms +
  bias → ParT's last layer; no out-projection, per-head scale, LayerNorms or MLP. Targets after the final LN:
  least squares **91.22 %**, tuned **93.29 %** (balanced dev; the head models: 93.2–93.4 %). Simpler and as good: the
  downstream MLP is not needed once the pooled terms are there. 'pre' variant (class token before the final LN,
  ParT's LN applied) running. If it holds: this is the cleanest ≥ 90 % model — per-particle formulas → 128 jet-level
  neurons → a linear layer — and goes through the full loop (one term per input, prune, 2M test, page).
- S15 pre (class token before the final LN, ParT's LN applied): 93.02 % → the post variant (93.29 %) is carried.
- User: S15 is the model to carry (simpler, no downstream MLP, no head values): one term per input → prune → re-tune →
  2M test → a page in the JEDI-linear neuron layout (128 neurons). α stays ParT's.

### S15 loop (clock ~11:45)
- One term per input per (neuron, head): least squares 87.41 % → re-tuned **92.85 %** (77,824 terms from 466,944).
- Pruning by (neuron, input), 8 rounds: 4,864 → **367 pairs**, 0–33 inputs per neuron, **92.77 %**. 2M test running.
- User (11:50): after S15, continue the α loop; and run the full loop on the 97 % model (formula α, ParT's values):
  one term per input, prune, tune toward ParT's probabilities, 2M test, page — "W1".
- S15p anatomy (dev): after pruning only **18 of the 128 neurons have particle inputs** (1, 17, 19, 25, 26, 53, 69, 71, 79, 82,
  84, 91, 98, 105, 116, 121, 124, 125); the other 110 are constants (their spread across jets 1e-4; ParT's last layer
  weighs the 18 with norm 7.7 vs 1.0 for the rest). 5,872 statements, 794 of them below 0.01 per jet → statement-level
  pruning (S15q) queued. **The class-token shares matter**: α_h,cls replaced by their means 75.3 %, the c·α_cls terms
  removed 47.8 % — but the particles' weights sum to 1 − α_cls, so part of that is normalization. Clean test queued (S16):
  the same fit with the particle weights renormalized to sum to 1 and no α_cls terms at all.
- S16 (S15 without the class-token share: particle weights renormalized to 1, no α_cls terms): least squares 87.96 %,
  tuned **91.03 %** (S15: 91.22 / 93.29 %). The 16 class-token shares are worth ≈ 2.3 pt; the model stands without them.
- S15q (statement-level pruning of S15p, 6 rounds, bisection within 0.1 pt): 5,872 → **5,020 statements, 92.86 %** (2M test running).
- A11 check: S15p's neuron formulas fed with the A2h formula α **as they are: 23 %** — the per-particle weights are close
  (mean |Δα| 0.018) but the class-token shares are not (corr 0.39 with ParT's) and S15p leans on them. → A11 refits the
  neuron formulas on the formula-α pooled terms (stage 1), then tunes scores, α_cls formula and neurons together (stage 2).
- P1 (bypass: the 12 hardest + 6 most displaced particles in fixed order, hinge terms + jet-level inputs → 128 neurons,
  no attention) started.
- User (12:05): the model meant is the 97 % formula α **with formula values**, full loop; the ParT-values variant (W1) may be
  done too but its page must be about what goes into each weight (the score formulas), not values.

### W1 loop, A11, P1 (clock ~12:15)
- **W1o**: the A2h selection formulas with at most one term per input per head: 97.04 → 96.54 % (least squares, 5,024
  statements from 29,952 coefficients) → re-tuned **96.97 %**. **W1p**: pruned by (head, input), 5 rounds, 5,024 →
  **3,664 pairs** (77–297 inputs per head), **96.85 %**. Rank agreement of the formula weights with ParT's: block-1 heads
  high, block-2 heads τ ≈ 0.16–0.18 — the block-2 selection is barely reproduced, and ParT's values carry it anyway.
  Page builder `alpha_page.py` (per head: what goes into the weight, every statement a drop-down, formula vs ParT's
  weights on example jets). 2M test evaluator for formula-α models to be written.
- **A11** (the formula α with formula values, direct structure): S15p's neurons with the formula α as they are 21.8 %;
  neurons refit on the formula-α pooled terms 74.7 %, tuned 77.1 %; everything tuned together **78.9 %**. Below the
  per-head all-formula A10 (84.1 %): the direct structure, linear in the pooled terms, cannot absorb the selection's errors
  the way ParT's values do (97 %) — the values' context is what compensates a wrong selection.
- **P1** (no attention at all: the 12 hardest + 6 most displaced particles in fixed order, hinge terms + jet-level inputs
  → 128 neurons): least squares ~, tuned **76.0 %**. A fixed ordering is a worse selection than the jet-level formulas (80 %).
- **S15q on the 2M test jets: 92.64 %** (accuracy 0.8412, AUC 0.9838; ParT 0.86 / 0.987) — 5,020 statements, 18 active neurons.

### A12 — which side needs the context? (clock ~12:35)
ParT's class attention run with keys or values replaced by ridge predictions from k-hop physics (A8's inputs, dev 10k):
| inputs | values predicted (ParT's α) | keys predicted (ParT's values) | both |
|---|---|---|---|
| own (38) | 72.5 % | **92.8 %** | 60.9 % |
| + kernel hop (126) | 78.4 % | 94.7 % | 71.6 % |
| + 2nd hop, pT-weighted (302) | 80.4 % | 95.2 % | 74.3 % |
| + 3 plain hops (356) | 80.7 % | 95.2 % | 75.1 % |
→ **The selection is the easy side: linear keys from a particle's own physics already give 92.8 % with ParT's values;
the values (what a head reads off a particle) are the side that needs ParT's context** — 80 % even with 3 hops, and
that is where the all-formula models lose (A10 84 %, A11 79 %). The two > 90 % lines are complementary: S15/S15q
(formula content, ParT's selection, 92.6 % test) and W1 (formula selection, ParT's content, 96.9 % dev). A fully
formula model needs formula VALUES with context — the per-particle embedding — which 3 hops of physics do not supply.
- **W1q** (W1p pruned further, 0.25 pt per round, 8 rounds): 3,664 → **2,165 (head, input) pairs** (8–244 inputs per head),
  **96.74 %** (from 96.85 %). Some heads become very compact (8–20 inputs), others keep ~250; test after W1p's.
- Asymmetry in W1q: block-1 heads keep 8–94 inputs, block-2 heads 177–244 (each input contributes little; τ vs ParT
  ≈ 0.17). Ablation on dev: block-2 score formulas replaced by a plain average → **94.70 %** (class-token share still from
  the jet-level formula; constant share instead 94.60 %); block-1 formulas replaced by a plain average → 92.12 %. Block 2's
  query comes from block 1's output (jet-dependent), so a per-particle formula is the wrong shape for it — and it hardly
  matters with ParT's values. → **W2**: block-1 selection formulas only (block 2 a plain average), re-tuned from W1q (running).
- **W2** (block-1 selection formulas only, block 2 a plain average; re-tuned from W1q, 10 epochs): 94.70 → **95.95 %** with
  **434 head–input pairs** (block-1 heads 8–94 inputs; block 2 none). The simplest ≥ 95 % model so far: 8 score formulas
  + 16 jet-level class-token formulas, ParT's values. Test queued with W1q.
- User (12:50): S15p's statements look repeated (one per head on the same input) → S17: one statement per (neuron, input)
  shared by the heads, per-head coefficients (running); more particle groups (firing patterns of the statements, JEDI-linear
  style) with qualitative descriptions; neurons sorted by importance.
- **S17** (from S15q: one statement per (neuron, input) shared by all heads — kind and threshold chosen jointly by least
  squares over the heads' pieces — per-head coefficients kept): 72.1 % before tuning → **91.50 %** after 60 epochs
  (**367 statements**, 5,020 coefficients; S15q 92.86 % with 5,020 statements). Longer tune running. Pages: the one-hot
  particle-type inputs and the charge are combined into one lookup statement each (exact), neurons listed most important
  first, particle groups = the firing patterns of the most discriminating statements (top 10, with descriptions).
- S17 tuned 200 epochs: **92.27 %** (367 shared statements; S15q 92.86 % with 5,020). 2M test and page queued (batch 3 with
  S15q, S5rpc1 rebuilt, W2).

### S18 — the 10 class logits directly (user, ~13:10)
ParT's last layer is linear, so with ParT's α the class scores themselves are per-particle formulas:
logit_c = b_c + Σ_h [Σ_i α_hi f_hc(x_i) + c_hc α_h,cls] — 10 formulas instead of 128 neurons + a 10 × 128 matrix, same
function class. Fit on ParT's logits (least squares, then toward ParT's probabilities + λ·R on the logits), one term per
input per (class, head), pruned by (class, input), then single statements. Running. Pages: the 128-neuron page gets a
class tab per class listing the neurons that go into its score (W × spread, linked to their drop-downs) and the inputs
that drive it; neurons sorted by importance everywhere.
- (13:40) Combined study **C1** (W1q selection formulas feeding the S15 neuron formulas; only ParT's last layer kept) running
  the full loop (fit → one term → prune → statements → 2M test). Try-a-jet page (weights ParT/formula × content
  ParT/formula, per-head and per-neuron drop-downs) builds with publish batch 3 after S17's test.
- (user, ~14:00) "The class blocks are sequential — why fit both?" The class token is a residual stream (c2 = c1 + block 2's
  read; both blocks read the same, unchanged particles), so the final 128 neurons contain both blocks' reads; S15/S17/S18
  fit only that final output, summing over the 16 heads' selections. **S19** tests one block's selection only (block-2
  heads / block-1 heads, 128 neurons, same fit as S15): running. C1o (combined, one term per input): 65.21 % (C1 78.35 %).

### 14:00–15:30
- **S17 on the 2M test: 92.13 %** (accuracy 0.8386, AUC 0.9832) — 367 statements, one per (neuron, input), shared by the heads.
- **S19** (one block's selection only, 128 neurons, same fit as S15, dev): block 1's 8 heads **90.51 %**, block 2's 8 heads
  88.89 % (both blocks 93.29 %). The class token is a residual stream; block 1's read is part of the output, and block 2's
  query depends on block 1, so neither block alone carries it; block 1 alone is the better half.
- **S18** (the 10 logits directly): fit 93.29 % (least squares 91.22 % — identical to S15's because least squares commutes
  with ParT's linear last layer); one term per input per (class, head) **92.49 %** (6,080 terms); pruned by (class, input)
  **241 pairs, 92.62 %**. Statement pruning (S18q) failed on an argument-parsing bug — fixed, rerunning, then test + page.
- **C1** (W1q selection formulas + formula neurons, only ParT's last layer): fit 78.35 %; one term 65.21 %; pruned 525 pairs
  64.95 %; statements 8,400 → 5,355, **70.06 %** (re-tuning recovered some). 2M test running (slow: formula selection).
- Published batch 3 (14:45): S17, S15q, S5rpc1 (merged type/charge statements, firing-pattern groups, importance order),
  W2, and the **Try-a-jet** page (jet/).
- **S18q** (the 10 class scores directly, statements pruned: 3,856 → 3,477, dev 92.69 %): **92.37 % on the 2M test jets**
  (accuracy 0.8397, AUC 0.9835). Ten formulas — one per class, one f per head — replace the 128 neurons and ParT's last layer.
- **W1p on the 2M test jets: 96.75 %** (accuracy 0.8561, AUC 0.9870; ParT 0.861 / 0.987) — the selection written as formulas
  (3,664 head–input pairs, one term per input), ParT's values: within 3.3 pt of ParT and its accuracy within 0.5 pt.
- **Best halves plugged together unchanged** (dev 20k): S15 neurons + W1q weights 16.2 %; with ParT's class-token share
  and W1q's particle weights 70.9 %; S16 neurons (no class-token term) + W1q particle weights 39.3 % (91.0 % with ParT's).
  The W1q weights pick the same top particle as ParT in only **10–48 % of jets per head** — yet with ParT's values they
  give 96.75 %: ParT's values are redundant enough that many different weightings pool to the same answer; the formula
  neurons are not. So the halves must be fitted to each other (C1 78 %, C2 running), and the share term is worth ~55 pt
  of the mismatch.

### Combined model is now the priority (user, ~17:40)
Why 96 % + 93 % → 78 %: each half was fitted against ParT's other half; the halves are not independent (W1q's weights are
functionally right for ParT's values but different in detail; formula neurons are not robust to that). Plan:
- C2: from C1q (5,355 statements), selection + neuron formulas tuned together (30k jets): 70.1 → 74.6 % after 10 epochs, running.
- C3: the combined fit with 100k jets (C1 had 30k for ~860k coefficients) and 60 joint epochs, dense — the ceiling of the
  combined structure; then the loop (one term, prune, statements) with joint re-tuning at each step; then the 2M test.
- **C2** (from C1q, selection + neuron formulas tuned together, 30k jets): 70.06 → **75.18 %** with 5,355 statements.
- **C3** (combined, 100k fitting jets, 60 joint epochs, dense): stage 1 78.44 % → **82.01 %** (C1 with 30k jets: 78.35 %).
  2M test of C3 started; the loop on C3 (one term → prune → statements, selection fixed) followed by a joint re-tune of both
  halves (C3j, 40 epochs) and its 2M test. C4 (neurons fitted toward W1q's own neurons): stage 1 78.41 % — no better than
  toward ParT's (78.44 %); stage 2 running.
- (user, 18:30) Terms per input: the one-term rule costs the combined model 13 pt (C3 82.01 → C3o 68.97 %; S15 lost 0.4 with
  ParT's weights). The user lifted it: **C6** = C3 (unrestricted, 82.01 %) → prune (neuron, input) pairs → prune single
  statements → joint re-tune of both halves → 2M test. (A 2-term variant, C5, was started and stopped in favour of C6.)
  Cancelling term pairs will be checked on the result.
- **C4** (combined, neurons fitted toward W1q's own neurons — ParT's values pooled with the formula weights — instead of
  ParT's, 100k jets, 60 joint epochs): **82.05 %** (C3, toward ParT's neurons: 82.01 %). The target makes no difference:
  the combined structure plateaus at ≈ 82 % on dev however the neurons are targeted.
- C3o one term 68.97 %; C3p pairs pruned (summed single costs) 4,864 → 361 in one round, 66.55 % — overshoot → C6's
  pair pruning switched to bisection on the measured agreement (prune_pairs).
- **W1q on the 2M test jets: 96.65 %** (accuracy 0.8557, AUC 0.9870) with 2,165 head–input pairs; **W2: 95.77 %** (accuracy
  0.8524, AUC 0.9863) with only the 8 block-1 selection formulas (434 pairs), block 2 a plain average.
- (19:30) Memory pressure (11 % free): the C1q 2M test (75 % done; C1q superseded by C3/C6) stopped → 21 % free.
- C6 statements: 74,208 → 59,632 in 5 rounds at ≈ 82.6 % — the bisection limit (0.1 pt per round) removes little per
  round; an L1 penalty during re-tuning or a target-agreement prune would cut deeper (next).
- C3 (one term) loop: C3q 4,504 statements, 74.05 % (then C3j joint re-tune).
- **C7** (combined; the neuron formulas read 126 inputs — + one hop of pair-kernel context —, standard basis, 60k jets): stage 1
  least squares 76.48 % (C3 with 38 inputs and 100k jets: 74.72 %), after neuron tuning 78.65 % (78.44 %), joint **80.55 %**
  (C3 82.01 %). Context helps the least-squares fit but not the tuned result here (fewer jets, 12k terms per neuron).
- **C6q**: 57,010 statements, 82.55 %. Next: C6j (joint re-tune) + 2M test; then **C6s** = L1 sparsification of C6j to the
  fewest statements with ≥ 81 %, and its 2M test.
- **C3j** (the one-term chain: C3 → one term 68.97 → pairs 66.55 → statements 74.05 → both halves re-tuned together):
  **78.83 %** with 4,504 statements, one term per input. 2M test queued in its chain. The no-limit chain (C6) keeps ≈ 82.5 %
  with 57,010 statements; the joint re-tune (C6j) and L1 sparsification (C6s) are next.
- **C6j** (joint re-tune of the pruned C6): 82.53 % — no gain; 2M test running. C6s's first run had a bug (the L1-penalized
  weights were discarded for the best-agreement checkpoint, i.e. the start) → only the size threshold acted (57,010 → 46,959
  statements, 82.55 %); fixed and rerunning. The 2M tests of the unpruned C3 and of the buggy C6s were stopped (superseded,
  machine load); running: C6j and C3j tests.
- **C6s, L1 path** (from C6j, 57,010 statements, 82.53 %; each strength: penalized re-tune, statements < 1 % of the largest
  dropped, re-tuned): 1e-3 35,739 / 82.75 % · 3e-3 20,404 / 82.52 % · **1e-2 8,267 / 81.49 %** · 3e-2 5,886 / 80.31 % ·
  1e-1 5,474 / 79.74 %. Kept for the 2M test: 8,267 statements, 81.49 % (target ≥ 81 %). At equal size the no-limit route
  beats one term per input: 5,886 statements → 80.3 % vs C3j 4,504 → 78.8 %.
- (22:05) The C6j 2M test (40 % done) stopped so the C6s and C3j tests finish sooner — C6s (8,267 statements) is the
  combined model that gets a page; C6j (57,010 statements) is the unreadable reference.
