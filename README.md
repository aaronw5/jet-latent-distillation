# Distilling the Latent Space of a Real-Time Jet Tagger into Interpretable Physics Equations

The 16 numbers of the JEDI-linear jet tagger's last hidden layer ("jet layer 4"), written as sums of if-statements on jet
physics observables; the network's own last layer turns them into the class scores (g, q, W, Z, t).

Site: https://aaronw5.github.io/jet-tagger-distillation/

*Work in progress: the code is being moved here from the analysis repository part by part.*

## Layout
| package | what it does |
|---|---|
| `jetdistill/observables` | the jet observables (numpy for fitting, plain Python for the exported files; tested equal) |
| `jetdistill/mars` | step 1: MARS, each network neuron as a sum of hinges on observables |
| `jetdistill/tuning` | step 2: all coefficients tuned together through the network's last layer (+ neuron fidelity λ); step 3: pruning |
| `jetdistill/simplify` | step 4: smaller formulas (optional) |
| `jetdistill/export` | stand-alone Python files of every formula, checked against the formula |
| `jetdistill/explain` | data packs for the written explanations and their checks |
| `jetdistill/site` | the pages and the site |
| `jetdistill/analysis` | mass experiments, untrained-network control |
| `configs/` | one file per setup (observables, thresholds, tuning target) |
| `scripts/` | run one setup end to end, publish the site |
