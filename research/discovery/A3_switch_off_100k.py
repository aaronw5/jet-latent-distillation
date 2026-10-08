"""A3 — do the attention-weights fit's class-token-share formulas (W1q) reproduce ParT's switch-off, on 100,000 validation
jets? Per head: the formula's and ParT's share by true class (± standard error), their correlation across jets, and for
b2h7 the mean contribution of each of its largest terms to u by class (α_cls = 1 / (1 + e^u))."""
import json, numpy as np
from jetdistill.pipeline import jets
from jetdistill.part.network import ParTNetwork
from jetdistill.research.heads import extract, rows_of_split, OUT
from jetdistill.research.direct_alpha import zfun
from jetdistill.config import CLASSES as CL
n = 100000; model = ParTNetwork('full').model; model.eval(); J = jets('full', 'dev'); rows = rows_of_split('dev', n)
_, A, M, L, _ = extract(model, 'dev', n, 'mps'); y = np.asarray(J['y'][rows])
Z = zfun(model)(J, rows, M); PA = np.load(OUT / 'W1q_alpha.npz'); cz = PA['cz']; u = Z @ cz.T; share_f = 1 / (1 + np.exp(u))
keys = [k for k in jets('full', 'fit')['Q']]
from jetdistill.research.ceiling_jet import matrix
Q100 = matrix(jets('full', 'fit'), keys)[:100000]; okq = np.isfinite(Q100).all(0) & (Q100.std(0) > 0); ZN = [k for k, o in zip(keys, okq) if o] + ['ln n particles', '1']
out = dict(n_jets=n, heads={})
for b, h in ((2, 7), (2, 4), (2, 8)):
    hi = 8 * (b - 1) + h - 1; sp = A[b - 1, :, h - 1, 0]; sf = share_f[:, hi]
    out['heads'][f'b{b}h{h}'] = dict(formula=[(float(sf[y == c].mean()), float(sf[y == c].std() / np.sqrt((y == c).sum()))) for c in range(10)],
                                     part=[(float(sp[y == c].mean()), float(sp[y == c].std() / np.sqrt((y == c).sum()))) for c in range(10)], corr=float(np.corrcoef(sf, sp)[0, 1]))
    print(f'b{b}h{h}: corr {out["heads"][f"b{b}h{h}"]["corr"]:.2f}', {CL[c]: (round(100 * out['heads'][f'b{b}h{h}']['formula'][c][0]), round(100 * out['heads'][f'b{b}h{h}']['part'][c][0])) for c in range(10)})
hi = 14; contrib = Z * cz[hi][None]; top = np.argsort(-np.abs(contrib).mean(0))[:6]
out['b2h7_terms'] = [dict(name=ZN[t], coef=float(cz[hi, t]), by_class=[float(contrib[y == c, t].mean()) for c in range(10)]) for t in top]
out['b2h7_u_by_class'] = [float(u[y == c, hi].mean()) for c in range(10)]
for t in out['b2h7_terms']: print(t['name'], round(t['coef'], 2), {CL[c]: round(v, 2) for c, v in enumerate(t['by_class'])})
(OUT / 'A3_switch_off_100k.json').write_text(json.dumps(out)); print('saved')
