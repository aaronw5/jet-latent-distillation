#!/bin/zsh
# JEDI-linear: at most one term per observable per neuron (n8 now; n64 after its step 1 + tuning)
cd /Users/anrunw/Documents/jet-latent-distillation
export KERAS_BACKEND=jax JETDISTILL_MODELS=/Users/anrunw/Documents/jedi-distill/context/JEDI-linear/official_models
PY=/Users/anrunw/Documents/distill/.venv/bin/python
date "+START one_term n8 %T" >> logs/one_term.log
$PY -u -m jetdistill.one_term all 8 805 399 2>&1 | grep -v -i warn >> logs/one_term.log
date "+START pipeline n64 step1 tune %T" >> logs/one_term.log
$PY -u -m jetdistill.pipeline all 64 step1 tune 2>&1 | grep -v -i warn >> logs/one_term_n64.log
T=$($PY -c "import json; r=json.load(open('results/all/n64/formulas.json')); print(next(t for t,m in r.items() if m['family']=='network' and m['role']=='main'))")
date "+START one_term n64 $T %T" >> logs/one_term.log
$PY -u -m jetdistill.one_term all 64 $T 2>&1 | grep -v -i warn >> logs/one_term.log
date "+DONE %T" >> logs/one_term.log
