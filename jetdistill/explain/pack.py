"""Data packs from which the explanation texts are written (by an AI agent; tools in check.py test the texts against them).

pack.json, per neuron: its if-statements (normalized shares, with what each observable means), its values per true class
(distribution, mean, on-rate, AUC against the rest), how much each class score relies on it, the observables it follows
most closely (rank correlation), and the validation accuracy lost when it is frozen at its mean; `network_match`: how
well it matches the network's own neuron on the whole test file (correlation, R², on/off agreement, 30×30 histogram).
Per class score: the neurons it relies on, its values per true class, the observables it follows.
Jets: 60,000 random training jets (not validation jets) for the descriptions, the validation jets for the frozen
accuracy, the whole test file for the network match."""
import numpy as np
from sklearn.metrics import roc_auc_score
from .. import formula as F
from ..config import CLASSES as CL, NC, RELU, TAGGER
from ..network import round_wrap
from ..observables import library

HOW = ('16 neurons; each neuron z = intercept + sum of if-statement terms of jet quantities, h = max(0, z) rounded; class score_c = B_c + '
       'sum_j W_jc h_j; class = largest score. "share" = fraction of the neuron\'s average input from that if-statement (sign: pushes the neuron '
       'up/down); weight_share = signed fraction of the class score\'s average input from that neuron.')
CLASS_TEXT = 'g = gluon jet, q = light-quark jet, W / Z = boosted W / Z boson (2 prongs, mass ~80 / 91 GeV), t = boosted top quark (3 prongs, mass ~173 GeV)'
if TAGGER == 'part':
    HOW = ('128 neurons (the class token after the last LayerNorm); each neuron h = intercept + sum of if-statement terms of jet quantities '
           '(no activation, no rounding; h can be negative); class score_c = B_c + sum_j W_jc h_j; class = largest score. "share" = fraction of the '
           'neuron\'s average input from that if-statement (sign: pushes the neuron up/down); weight_share = signed fraction of the class '
           'score\'s average input (average |h_j| times W_jc) from that neuron.')
    CLASS_TEXT = ('QCD = light-quark or gluon jet; Hbb, Hcc, Hgg = Higgs boson (mass ~125 GeV) decaying to b b̄, c c̄, g g (2 prongs); H4q = H → WW* → 4 quarks '
                  '(4 prongs); Hqql = H → WW* → lepton, neutrino, 2 quarks; Zqq, Wqq = Z / W boson to 2 quarks (mass ~91 / 80 GeV); '
                  'Tbqq = top quark (mass ~173 GeV) to b and 2 quarks (3 prongs); Tbl = top quark to b, lepton, neutrino')


def neurons_and_logits(f, Q, last):
    """h (the neurons after the activation and the network's rounding, if any) and the class scores"""
    K, b, i_, f_ = last; H = round_wrap(F.hidden(f, Q), i_, f_); return H, H @ K + b


def _test(t):
    p = lambda q, k, th: f"{q} {'>' if k == 'gt' else '<' if k == 'lt' else '(linear)'} {th if k != 'lin' else ''}".strip()
    return p(t['q'], t['kind'], t.get('t')) + (f" AND {p(t['q2'], t['kind2'], t.get('t2'))}" if t.get('q2') else '')


def scale_facts(h, y, shares=None, nb=30):
    """a neuron (or class score) as a scale, with its readings as FACT text computed from the numbers"""
    lo, hi = float(h.min()), float(np.quantile(h, .995)); hi = hi if hi > lo else lo + 1e-6; e = np.linspace(lo, hi, nb + 1)
    means = {CL[c]: float(h[y == c].mean()) for c in range(NC)}; auc = {CL[c]: float(roc_auc_score(y == c, h)) if np.ptp(h) > 0 else 0.5 for c in range(NC)}
    order = sorted(CL, key=lambda c: -means[c]); best = max(CL, key=lambda c: abs(auc[c] - 0.5))
    out = dict(values_by_true_class=dict(edges=[round(float(v), 4) for v in e], zero_share={CL[c]: round(float((h[y == c] <= lo).mean()), 3) for c in range(NC)},
                                         per_class=[[round(float(v), 4) for v in np.histogram(np.clip(h[y == c], lo, hi), e)[0] / max((y == c).sum(), 1)] for c in range(NC)]),
               auc_vs_rest={c: round(v, 3) for c, v in auc.items()}, class_order_by_mean=order,
               FACT_scale='largest for ' + ', then '.join(f'{c} ({means[c]:.2f})' for c in order) + f'; it separates {best} jets from the rest best '
                          f"(AUC {auc[best]:.2f}: {'large' if auc[best] > 0.5 else 'small'} for {best})")
    if shares is not None:
        up = [f'{CL[c]} ({100 * shares[c]:+.0f}%)' for c in range(NC) if shares[c] > 0.02]; dn = [f'{CL[c]} ({100 * shares[c]:+.0f}%)' for c in range(NC) if shares[c] < -0.02]
        no = [CL[c] for c in range(NC) if abs(shares[c]) <= 0.02]
        out['FACT_used_by'] = ('raises the score of ' + ', '.join(up) if up else '') + ('; ' if up and dn else '') + ('lowers the score of ' + ', '.join(dn) if dn else '') + \
                              (f"; does not (or hardly) enter the score of {', '.join(no)}" if no else '') + ' (share of each class score’s average input)'
    return out


def build(f, normalized, n, last, jets, dev, title=''):
    """jets / dev: dict(Q=observables, y=true class) — the 60,000 explanation jets and the validation jets"""
    desc = {k: o.desc for k, o in library(n).items()}; Q, y = jets['Q'], jets['y']; N_, C_ = normalized
    H, Lg = neurons_and_logits(f, Q, last); Hd, Lgd = neurons_and_logits(f, dev['Q'], last); acc0 = float((Lgd.argmax(1) == dev['y']).mean())
    ranks = {q: np.argsort(np.argsort(v)) for q, v in Q.items()}

    def follows(v, k=6):
        if np.ptp(v) == 0: return []
        rv = np.argsort(np.argsort(v)); cs = [(abs(c), q, c) for q in ranks for c in [np.corrcoef(rv, ranks[q])[0, 1]] if np.isfinite(c)]
        return [dict(quantity=q, meaning=desc.get(q, q), rank_corr=round(float(c), 3)) for _, q, c in sorted(cs, reverse=True)[:k]]
    neurons = []
    for nr in N_:
        j = nr['neuron']; h = H[:, j]; Hf = Hd.copy(); Hf[:, j] = Hd[:, j].mean(); frozen = float(((Hf @ last[0] + last[1]).argmax(1) == dev['y']).mean())
        sh = [float(C_['share'][j][c]) for c in range(NC)]
        neurons.append(dict(neuron=j, importance_share=round(float(C_['neuron_importance'][j]), 4), **scale_facts(h, y, sh),
                            weight_share_in_each_class_score={CL[c]: round(sh[c], 3) for c in range(NC)},
                            on_rate_by_true_class={CL[c]: round(float((h[y == c] > 0).mean()), 3) for c in range(NC)},
                            mean_value_by_true_class={CL[c]: round(float(h[y == c].mean()), 3) for c in range(NC)},
                            accuracy_drop_if_frozen_points=round(100 * (acc0 - frozen), 3), follows_quantities=follows(h),
                            if_statements=[dict(test=_test(t), share_of_neuron_input=round(float(t['share']), 3),
                                                meaning=desc.get(t['q'], t['q']) + (f"; {desc.get(t['q2'], t['q2'])}" if t.get('q2') else '')) for t in nr['terms'][:14]]))
    scores = []
    for c in range(NC):
        sh = sorted([(abs(C_['share'][j][c]), j, C_['share'][j][c]) for j in range(len(N_))], reverse=True)
        scores.append(dict(cls=CL[c], **scale_facts(Lg[:, c], y), relies_on_neurons=[dict(neuron=j, share=round(float(s), 3)) for _, j, s in sh if abs(s) > 0.02],
                           mean_score_by_true_class={CL[k]: round(float(Lg[y == k, c].mean()), 3) for k in range(NC)}, follows_quantities=follows(Lg[:, c])))
    return dict(n_particles=n, formula_terms=F.tag(f), title=title, accuracy_validation=acc0, classes=CLASS_TEXT, how_the_formula_works=HOW,
                neurons=neurons, class_scores=scores)


def network_match(f, Q, H_net, last, H=None):
    """per neuron: the formula's neuron (after rounding) against the network's own neuron, on the given jets (H: the
    formula's neurons, when already computed)"""
    H = neurons_and_logits(f, Q, last)[0] if H is None else H; out = {}
    for j in range(H.shape[1]):
        a, b = H[:, j], H_net[:, j]; hi = float(max(np.quantile(a, .998), np.quantile(b, .998), 1e-6))
        lo = 0.0 if RELU else float(min(np.quantile(a, .002), np.quantile(b, .002))); lo = min(lo, hi - 1e-6); e = np.linspace(lo, hi, 31)
        out[j] = dict(corr=round(float(np.corrcoef(a, b)[0, 1]), 3) if a.std() > 0 and b.std() > 0 else None, r2=round(float(1 - ((a - b) ** 2).mean() / max(b.var(), 1e-12)), 3),
                      onoff=round(float(((a > 0) == (b > 0)).mean()), 3) if RELU else None, mean_formula=round(float(a.mean()), 3), mean_network=round(float(b.mean()), 3), n_jets=int(len(a)),
                      hist2d=dict(edges=[round(float(v), 4) for v in e], counts=np.histogram2d(np.clip(b, lo, hi), np.clip(a, lo, hi), [e, e])[0].astype(int).tolist()))
    return out


COMBOS_HOW = ('For each neuron: ALL its if-statements ("tests", strongest first) and its main GROUPS of jets (every jet is in one group) '
              'formed by how the neuron value is built (k-means on what each if-statement adds; no labels used). Per group ("patterns"): '
              'share of jets, true-class mix (g,q,W,Z,t; looked up afterwards), the class the formula decides (mix) and how often it is '
              "right, mean neuron value and how often the neuron is on, per if-statement: pass_rate (share of the group's jets for which "
              'it passes) and added_by_each (mean amount it adds in this group; negative = lowers the neuron), how much the neuron adds '
              'to each class score (g,q,W,Z,t), mean mass/width/sum_pt, mean pT [GeV] of the hardest particles (1st, 2nd, ...), share '
              'of pT at each distance from the axis (radial_bins). The neuron = max(0, intercept + sum of what the if-statements add), '
              "rounded to the network's number grid, which holds values from 0 up to just under largest_value: a larger sum WRAPS AROUND "
              '(value modulo largest_value), as in the network, so a group whose sum is far above largest_value has an effectively scrambled value.')
COMBOS_KEEP = ('frac', 'classes', 'formula_decides', 'formula_right', 'mean_neuron', 'neuron_on', 'pass_rate', 'added_by_each', 'adds_to_scores', 'means', 'pt_rank', 'radial')


def combos_pack(anatomy, explain, pack, last):
    """the data for the texts on each neuron's main groups of jets (needs the neuron texts, explain.json, first)"""
    r4 = lambda v: [round(x, 4) for x in v] if isinstance(v, list) else round(v, 4) if isinstance(v, float) else {k: round(x, 4) for k, x in v.items()} if isinstance(v, dict) else v
    pk = {n['neuron']: n for n in pack['neurons']}
    out = dict(formula=anatomy['formula'], n_particles=pack['n_particles'], classes=CL, all_jets={k: r4(anatomy['all'][k]) for k in ('classes', 'means', 'pt_rank', 'radial') if k in anatomy['all']},
               radial_bins=anatomy['radial_bins'], how_to_read=COMBOS_HOW, neurons=[])
    for n in anatomy['neurons']:
        j = n['neuron']; x = explain['neurons'].get(str(j), {})
        out['neurons'].append(dict(neuron=j, intercept=round(n['intercept'], 4), largest_value=float(2 ** last[2][j]) if last[2] is not None else None, scale=x.get('title', ''), measures=x.get('measures', ''),
                                   FACT_scale=pk[j].get('FACT_scale', ''), tests=n['combo_tests'], patterns=[{k: r4(c[k]) for k in COMBOS_KEEP if k in c} for c in n['combos']]))
    return out
