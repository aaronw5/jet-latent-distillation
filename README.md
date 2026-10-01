# ParT latent distillation — built pages

Built from the branch `part-tagger-mac` (JetClass; ParT_full and ParT_kin), with `python -m jetdistill.pipeline ... page`.
- `all/full/index.html` — ParT_full, setup "all": formulas 1658 (17 neurons, the smallest without loss), 887 (13 neurons,
  pruned), 703 (step 4)
- `all/kin/index.html` — ParT_kin, setup "all": formulas 1938 (20 neurons, no loss), 923 (15 neurons), 736 (step 4)
Results are in the ParT paper's metrics on 2M test jets (10 % of the JetClass test set). Each page loads its
`explain_*.json` from the same folder: serve this branch with GitHub Pages, or download it and open locally.
