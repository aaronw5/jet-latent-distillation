"""The numpy observables equal the plain-Python code of the exported files (needs the dataset, see README)."""
import numpy as np
import pytest
from jetdistill import data
from jetdistill.observables import library, compute, compile_quantities


@pytest.mark.parametrize('n', [8, 64])
def test_numpy_equals_python(n):
    x = data.particles('test', n, stop=100)
    ids = list(library(n)); O = compute(x, n); q = compile_quantities(ids, n)
    P = [q(r[:, 0], r[:, 1], r[:, 2]) for r in x]
    for k in ids:
        b = np.array([getattr(p, k) for p in P], float)
        assert np.allclose(O[k], b, rtol=1e-6, atol=1e-9), k


def test_mass_rule():
    from jetdistill.observables import mass_ids
    m = mass_ids(8)
    assert {'mass', 'm01', 'm012', 'sd_mass', 'mass_over_sum_pt'} <= m and 'tau21' not in m
