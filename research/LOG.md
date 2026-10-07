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
