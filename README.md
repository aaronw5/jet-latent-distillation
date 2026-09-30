# Distilling the Latent Space of a Real-Time Jet Tagger into Interpretable Physics Equations

The JEDI-linear jet tagger (a small fixed-point network for FPGA triggers) sorts particle jets into gluon (g), light quark
(q), W, Z and top (t). It decides from 16 numbers, its last hidden layer. This repository writes each of those 16 numbers as a
sum of if-statements on jet physics observables (jet mass, ΣzΔR² = Σᵢ pTᵢΔRᵢ² / Σᵢ pTᵢ, N-subjettiness, energy correlations, the hardest
particles, …) and feeds them to the network's own, unchanged last layer. The result is a formula that makes the network's
decision on about 9 of 10 jets (8 particles) and 19 of 20 jets (64 particles), and a stand-alone Python file of it.

Site: https://aaronw5.github.io/jet-tagger-distillation/

## Results (setup "All observables", whole test file, 260,000 jets)

| network | formula | if-statements | same class as the network | accuracy (network) |
|---|---|---:|---:|---:|
| 8 particles | tuned on the network | 782 | 90.54 % | 65.65 % (65.69 %) |
| 8 particles | smaller version | 399 | 90.09 % | 65.54 % |
| 64 particles | tuned on the network | 690 | 94.18 % | 81.37 % (80.94 %) |
| 64 particles | smaller version | 250 | 93.70 % | 81.18 % |

**Control.** The same step 1 on an untrained copy of the network (random weights at each layer's trained scale, the same
number formats) explains a median 16 % of each neuron's variance at 64 particles (55 % at 8), against 92 % (89 %) for the
trained network. Boosted trees on the raw particle inputs explain 79 % (83 %) of the trained neurons: less than the physics
observables do. Step 1 on the raw inputs only explains 42 % (58 %).

A run of this repository (`python -m jetdistill.pipeline all 8`, 1.2 hours) reproduces the 8-particle numbers: 805
if-statements, 90.57 % / 65.66 %; smaller version 399, 90.06 % / 65.56 %; step 1 identical. Formula sizes can differ
slightly between runs, since pruning keeps any cut that stays within 0.1 point of validation agreement.

Nothing is fitted or chosen on the test file: fitting uses jets of the training archive, choices (lengths, sizes) use
validation jets of the training archive, and the validation archive of the dataset is the test file.

## Method

1. **Step 1 — if-statements per neuron** (`mars.py`). Each neuron's value before its ReLU is fitted by forward selection
   (MARS, Friedman 1991) over terms `Q`, `max(0, Q − t)`, `max(0, t − Q)` and products of two, with `t` at the 5–95 %
   quantiles of each observable. Only the network's neuron values are used, never the classes.
2. **Step 2 — tuned together** (`tuning.py`). The first K terms of every neuron keep their thresholds; all coefficients
   are trained at once through the network's last layer (its fixed-point rounding included) toward the network's class
   probabilities, plus λ·R, R = the mean squared difference between each formula neuron and the network's neuron in
   units of its variance (λ = 0.01). Without R the 16 neurons can drift in directions the 5 class scores do not see.
3. **Step 3 — pruning** with retraining while validation agreement with the network stays within 0.1 point.
4. **Step 4 — smaller versions** (`simplify.py`, optional): terms and whole observables removed by weighted least
   squares on the neurons, thresholds refined and rounded; the smallest version within 0.5 point of validation agreement.

For comparison each setup also has formulas tuned on the **true labels** and formulas tuned only on the **network's neuron
values** (every step fitted to the neurons, no class scores).

**Setups** (`config.py`): all observables; no W/Z/H/top mass values offered as thresholds; no mass observables; no mass
observables or exact equivalents; all observables tuned toward the network's decisions (one-hot) instead of its
probabilities. **Control** (`analysis/control.py`): the same step 1 on an untrained network with random weights at the
trained scale.

## Install and data

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e .
# the hls4ml LHC jet dataset (150 particles per jet): hls4ml_LHCjet_150p_train.tar, hls4ml_LHCjet_150p_val.tar
python -m jetdistill.data hls4ml_LHCjet_150p_train.tar hls4ml_LHCjet_150p_val.tar      # -> data/
# the official JEDI-linear models: github.com/calad0i/JEDI-linear, official_models/ -> models/
```
Paths can be changed with `JETDISTILL_DATA`, `JETDISTILL_MODELS`, `JETDISTILL_RESULTS`.

## Run

```bash
python -m jetdistill.pipeline all 8          # every stage: step1 tune step4 export explain metrics page
python -m jetdistill.pipeline nomass 64 tune step4
python -m jetdistill.site.sites              # landing page and the setup selector (site/)
python -m jetdistill.analysis.control 8      # untrained-network control
python -m jetdistill.analysis.convergence 8  # step 2 for 3000 steps; training / validation / test agreement
python -m jetdistill.figures all             # the figures of the slides (results/figures/)
python slides/build_html.py                  # the slides as one HTML page (slides/deck.html)
JETDISTILL_SMOKE=1 python -m jetdistill.pipeline all 8    # every stage on a few thousand jets, in minutes
pytest tests
```
Stages write to `results/<setup>/n<N>/`: `step1.json`, `formulas.json` (the formulas of the setup, named by their
number of terms), `formulas/<tag>/` (formula, normalized weights, the two Python files and their check, explanation data),
`metrics.json`. Observables and network outputs of each set of jets are computed once per network (`results/_jets/`).

**Convergence.** Step 2 runs 800 Adam steps and keeps the step with the highest validation agreement. Run for 3000 steps
(8 particles, 100 if-statements per neuron), validation agreement is 90.53% at step 800 and 90.63% at step 3000, within
the ±0.15-point spread between checkpoints; training agreement rises by 0.2 points. The formulas give the same agreement on
the jets they were tuned on and on the test jets (e.g. 399: 90.25% training, 90.10% validation, 90.06% test).

**Names.** Pages, slides and texts write every quantity as its formula (ΣzΔR² = Σᵢ zᵢΔRᵢ², λ₁, τ₂₁(β=2), …; zᵢ = pTᵢ / ΣpT;
`observables/symbols.py`); the site's `quantities.html` lists all of them with their definitions. Code ids that are jargon
(girth, width, e2_sq) are replaced by descriptive names (sum_z_dr, lam1_plus_lam2, sum_zz_dr2) in the exported files.

## Written explanations

The page's texts for each neuron and class score (`formulas/<tag>/explain.json`) and for each neuron's groups of jets
(`combos.json`) are written from the explanation data by an AI agent following `docs/explanations.md`, and checked
against the numbers by `python -m jetdistill.explain.check` (a text may not contradict a sign or quote a number that is
not in the data, and names describe observables, never classes). Rerun `pipeline ... explain page` afterwards.

## Layout

| module | what it does |
|---|---|
| `observables/` | the observables: one table (`library.py`), numpy (`compute.py`), plain Python for the files (`python_code.py`) |
| `data.py`, `network.py` | the dataset and splits; the network layer by layer (identical to the model) |
| `mars.py`, `tuning.py`, `simplify.py` | steps 1, 2–3 and 4 |
| `formula.py` | a formula and its evaluation; `export.py`: the stand-alone Python files and their check |
| `explain/` | explanation data (`pack.py`, `anatomy.py`) and the text checks (`check.py`) |
| `metrics.py`, `pipeline.py` | metrics on the whole test file; the stages of one setup |
| `site/` | the page of a setup (`template.html`, `page.py`) and the site (`sites.py`) |
| `analysis/` | untrained-network control (`control.py`), step-2 convergence (`convergence.py`), jet-mass studies |
| `figures.py` | every figure of the slides, from the results |
| `slides/` (top level) | the talk: slide files, images, screenshot tool, a map from each image to what makes it (`slides/README.md`) |

Model: JEDI-linear (github.com/calad0i/JEDI-linear). Dataset: hls4ml LHC jet dataset (150 particles).
