"""A formula, its exported Python file and its normalized file give the same class on test jets; the chains of the
exported file equal the original terms exactly."""
import numpy as np
import pytest
from jetdistill import data, export, formula as F
from jetdistill.network import Network
from jetdistill.observables import compute

FORMULA = [dict(neuron=j, intercept=0.3 * (j % 3), terms=[dict(q='mass', kind='gt', t=60.0 + 5 * j, coef=0.01), dict(q='mass', kind='lt', t=120.0, coef=-0.02),
                                                          dict(q='tau21', kind='lin', t=None, coef=1.0 - 0.1 * j), dict(q='girth', kind='gt', t=0.05, q2='sum_pt', kind2='lt', t2=900.0, coef=0.002)])
           for j in range(16)]


@pytest.mark.parametrize('n', [8])
def test_files_equal_formula(n, tmp_path):
    net = Network(n); x = data.particles('test', n, stop=300); Q = compute(x, n, F.observables_used(FORMULA))
    ranges = {q: (float(v.min()), float(v.max())) for q, v in Q.items()}; pred = F.logits(FORMULA, Q, net.last).argmax(1)
    p1 = export.write_formula(FORMULA, n, net.last, ranges, 'test', 'test', tmp_path / 'f.py')
    p2 = export.write_normalized(FORMULA, n, net.last, export.normalize(FORMULA, Q, net.last), 'test', 'test', tmp_path / 'g.py')
    assert export.check(p1, x, pred)[0] == 1.0 and export.check(p2, x, pred)[0] == 1.0


def test_chains_exact():
    nr = FORMULA[3]; Q = {'mass': np.linspace(0, 300, 2001), 'tau21': np.zeros(2001)}
    ch = export.chains(dict(nr, terms=[t for t in nr['terms'] if not t.get('q2')]), {'mass': (0, 300), 'tau21': (0, 1)})
    seg = ch['pieces'][0]['segments']; z = np.full(2001, ch['intercept'])
    for lo, hi, sl, of in seg: z += np.where((Q['mass'] >= lo) & (Q['mass'] < hi) | ((hi == 300) & (Q['mass'] == 300)), sl * Q['mass'] + of, 0)
    ref = nr['intercept'] + sum(t['coef'] * F.basis(t, Q) for t in nr['terms'] if not t.get('q2'))
    assert np.allclose(z, ref)
