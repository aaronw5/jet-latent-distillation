"""A1 — ParT's class-attention weights on 100,000 validation jets, with standard errors (jets as the unit):
per head: weight on leptons / photons / charged (× an average particle), weight against track |d0|/σ, the class token's
share by true class, effective particles by true class. Saved to research/results/A1_attention_100k.json."""
import json, numpy as np
from jetdistill.pipeline import jets
from jetdistill.part.network import ParTNetwork
from jetdistill.part.clsfit import particle_features
from jetdistill.research.heads import extract, rows_of_split, OUT
from jetdistill.config import CLASSES as CL
n = 100000; model = ParTNetwork('full').model; model.eval(); J = jets('full', 'dev'); rows = rows_of_split('dev', n)
_, A, M, L, _ = extract(model, 'dev', n, 'mps'); y = np.asarray(J['y'][rows])
F0, ok = particle_features(J, rows, ctx=[]); npart = ok.sum(1)
d0s = np.abs(np.asarray(J['ext'][rows][..., 3], np.float32)) / np.maximum(np.asarray(J['ext'][rows][..., 4], np.float32), 1e-6)
lep = (F0[..., 11] + F0[..., 12]) > 0; pho = F0[..., 10] > 0; chg = F0[..., 7] != 0
bins = np.array([0, .5, 1, 2, 3, 5, 10, 1e9]); dbin = np.clip(np.searchsorted(bins, d0s, side='right') - 1, 0, 6)
def per_jet_ratio(rel, mask):
    """per jet: mean relative weight (α × n) of the masked particles; jets without such particles excluded"""
    cnt = (mask & ok).sum(1); s = np.where(mask & ok, rel, 0).sum(1); k = cnt > 0; v = s[k] / cnt[k]
    return float(v.mean()), float(v.std() / np.sqrt(k.sum())), int(k.sum())
out = {'n_jets': n, 'classes': CL, 'heads': []}
for b in range(2):
    for h in range(8):
        al = A[b, :, h, 1:]; rel = al * npart[:, None]; self_ = A[b, :, h, 0]
        eff = np.exp(-(np.where(ok, al / np.maximum(al.sum(1, keepdims=True), 1e-12), 1) * np.log(np.where(ok, al / np.maximum(al.sum(1, keepdims=True), 1e-12), 1))).sum(1))
        d = dict(head=f'b{b + 1}h{h + 1}', lepton=per_jet_ratio(rel, lep), photon=per_jet_ratio(rel, pho), charged=per_jet_ratio(rel, chg),
                 d0=[per_jet_ratio(rel, chg & (dbin == i)) for i in range(7)],
                 self_by_class=[(float(self_[y == c].mean()), float(self_[y == c].std() / np.sqrt((y == c).sum()))) for c in range(10)],
                 eff_by_class=[(float(eff[y == c].mean()), float(eff[y == c].std() / np.sqrt((y == c).sum()))) for c in range(10)])
        out['heads'].append(d); print(d['head'], 'lepton', round(d['lepton'][0], 2), '±', round(d['lepton'][1], 2), '| d0 3–5σ', round(d['d0'][4][0], 2), '±', round(d['d0'][4][1], 2), '| self max', CL[int(np.argmax([s for s, _ in d['self_by_class']]))])
(OUT / 'A1_attention_100k.json').write_text(json.dumps(out)); print('saved; jets per class', np.bincount(y, minlength=10).tolist())
