"""Paths and the setups (experiments).

Paths (environment variables, defaults relative to the working directory):
  JETDISTILL_DATA     data/     the dataset files made by `python -m jetdistill.data` (see README)
  JETDISTILL_MODELS   models/   the official JEDI-linear models (github.com/calad0i/JEDI-linear, official_models/)
  JETDISTILL_RESULTS  results/  everything the pipeline writes: results/<setup>/n<N>/...
"""
import os
from pathlib import Path

DATA = Path(os.environ.get('JETDISTILL_DATA', 'data'))
MODELS = Path(os.environ.get('JETDISTILL_MODELS', 'models'))
RESULTS = Path(os.environ.get('JETDISTILL_RESULTS', 'results'))
CLASSES = ['g', 'q', 'W', 'Z', 't']

# jets used by each step (all from the TRAIN archive except the test jets, which come from the VAL archive)
N_STEP1_FIT, N_TUNE_FIT, N_DEV, N_TEST_SPLIT = 40000, 100000, 25000, 50000

MASSES = {'m_W': 80.4, 'm_Z': 91.19, 'm_H': 125.1, 'm_t': 172.8}   # offered as candidate thresholds of mass observables
MASS_MULT = (.5, 1., 1.5, 2.)


class Setup(dict):
    __getattr__ = dict.get


SETUPS = {
    'all': Setup(label='All observables', page_desc='all jet quantities; candidate thresholds: the 5–95% quantiles of each quantity, plus the W/Z/H/top masses (and ×½, ×1.5, ×2) for mass quantities',
                 masses_as_thresholds=True),
    'nophys': Setup(label='No mass values as thresholds', page_desc='all jet quantities; candidate thresholds: only the 5–95% quantiles (no W/Z/H/top mass values offered)',
                    masses_as_thresholds=False),
    'nomass': Setup(label='No mass observables', page_desc='no quantity with units of mass (jet mass, masses of particle subsets, subjets, soft drop) and no ratio built from masses',
                    masses_as_thresholds=False, no_mass=True),
    'nomass_strict': Setup(label='No mass observables or exact equivalents', page_desc='as “No mass observables”, and also no quantity equal to (m/ΣpT)² in disguise (|correlation| > 0.98 on the training jets)',
                           masses_as_thresholds=False, no_mass=True, strict=True),
    'all_agree': Setup(label='All observables, tuned for agreement', page_desc='as “All observables”; step 2 is tuned toward the network’s decision for each jet (its class) instead of its probabilities',
                       masses_as_thresholds=True, step1_from='all', target='decisions', truth_family=False),
}
FIDELITY_LAMBDA = 0.01    # weight of the neuron-closeness term R in steps 2 and 3
