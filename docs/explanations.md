# Writing the explanation texts for one formula

You write the words for the "Neuron Interpretation" and "Class-Score Decomposition" tabs of one formula of the site
"Distilling the Latent Space of a Real-Time Jet Tagger into Interpretable Physics Equations". A formula = 16 neurons, each
a sum of if-statements (hinges `max(0, Q − t)`, `max(0, t − Q)`, products of two) on jet observables, fed into the
network's own last layer (5 class scores: g, q, W, Z, t). All numbers you need are in the data packs; never invent a number.

The formula's folder is `results/<setup>/n<N>/formulas/<TAG>/` (called DIR below); run `python -m jetdistill.pipeline <setup> <N> explain`
first (it writes pack.json and anatomy.json).

## Part 1 — `DIR/explain.json`
Read `DIR/pack.json` (its "how" field explains every number).
Structure: `{"formula": "<TAG>", "summary": "...", "neurons": {"0": {"title", "measures", "role", "importance"}, ... "15"},
"class_scores": {"g": {"title", "measures", "built_from"}, ... "t"}}` — all 16 neurons (a neuron that is always 0 gets a
one-line text saying so), all 5 classes.

Rules:
- Plain, academic, specific. No metaphors, no flourish, no "essentially", no "acts as". Say what is computed.
- Titles describe the physics the neuron measures in observable terms ("jet mass between 87 and 97 GeV, narrow jets",
  "wide jets with three hard subjets"). NO class names in neuron titles or group names, and never "-like"/"likeness"
  ("quark-like", "top-likeness"): classifying is the network's job, the text says what is measured.
- "measures": which if-statements dominate (quote thresholds as in the pack), what raises/lowers the neuron, the mean value
  per true class (numbers from the pack), how often it is zero.
- "role": which class scores it raises/lowers (sign and size of its weight share from the pack), only those it enters.
- Numbers: quote them exactly as the pack gives them (rounding to 2–3 significant digits is fine).
- If a neuron's values can exceed 2^i (its integer bits in the pack), mention that its largest values wrap around.

Check (must print no problems, exit 0):
`python -m jetdistill.explain.check texts DIR/pack.json DIR/explain.json`
`python -m jetdistill.explain.check nolabels DIR/explain.json`
Fix and rerun until both are clean.

## Part 2 — `DIR/combos.json`
Rerun `python -m jetdistill.pipeline <setup> <N> explain`: with explain.json present it writes `DIR/combos_pack.json`
(its "how_to_read" field explains it). Structure: `{"formula": "<TAG>", "neurons": {"<j>": {"<group id>": {"name", "jets",
"why"}, ...}, ...}}`, every group of every neuron in the pack (group id = its index). "name": what the jets in the group have in common, in observables (no class names, no "-like").
"jets": the class mix and the observable means from the pack. "why": which if-statements pass and what they add (from the
pack's pass_rate / added_by_each). If a group's summed additions exceed the neuron's range (largest_value, integer bits),
say that the value wraps around rather than claiming it is at its top.

Check (must be clean):
`python -m jetdistill.explain.check combos DIR/combos_pack.json DIR/combos.json`
`python -m jetdistill.explain.check numbers DIR/combos_pack.json DIR/combos.json`
`python -m jetdistill.explain.check nolabels DIR/combos.json`

Do not modify any other file.
