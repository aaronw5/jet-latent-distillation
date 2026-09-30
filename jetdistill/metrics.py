"""Accuracy, agreement with the network, and per class (one class against all the others): AUC, background rejection
1/FPR at 30 / 50 / 80 % signal efficiency (scanning the model's probability for the class), and signal efficiency and
background rejection at the model's own decision. On the whole test file."""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from .config import CLASSES, NC


def softmax(L):
    e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


# the metrics of the ParT paper's JetClass tables (Qu, Li & Qian 2022, section 2 and tables 1, 3): each signal class
# against QCD with score_S / (score_S + score_QCD), Rej = 1 / FPR at the given signal efficiency
PAPER_EFF = dict(Hbb=.5, Hcc=.5, Hgg=.5, H4q=.5, Hqql=.99, Tbqq=.5, Tbl=.995, Wqq=.5, Zqq=.5)


def paper_metrics(P, y):
    """accuracy, AUC (sklearn roc_auc_score, average='macro', multi_class='ovo', as the paper's footnote 3) and Rej_X%
    per signal class, the FPR taken at the first ROC point with TPR > X (weaver's bkg_rejection); n_bkg_pass = the
    number of QCD jets above that threshold (the rejection's statistical precision)"""
    out = dict(accuracy=float((P.argmax(1) == y).mean()), auc=float(roc_auc_score(y, P, average='macro', multi_class='ovo')), rej={})
    b = CLASSES.index('QCD')
    for c, eff in PAPER_EFF.items():
        s = CLASSES.index(c); m = (y == s) | (y == b); sc = P[m, s] / np.maximum(P[m, s] + P[m, b], 1e-300)
        f, tp, _ = roc_curve(y[m] == s, sc); i = int(np.argmax(tp > eff)); nb = int((y[m] == b).sum())
        out['rej'][c] = dict(eff=eff, rej=float(1 / f[i]) if f[i] > 0 else None, n_bkg_pass=int(round(f[i] * nb)), n_bkg=nb)
    return out


def metrics(L, y, ref=None):
    """L: class scores (J, classes); y: true class; ref: the network's class"""
    P = softmax(L); pred = L.argmax(1); out = dict(accuracy=float((pred == y).mean()), per_class={})
    if ref is not None: out['same_as_network'] = float((pred == ref).mean())
    if 'QCD' in CLASSES and len(np.unique(y)) == NC: out['paper'] = paper_metrics(P, y)      # needs every class (not in smoke runs)
    for c in range(NC):
        yy = y == c; f, tp, _ = roc_curve(yy, P[:, c]); keep = np.unique(np.linspace(0, len(f) - 1, min(len(f), 150)).astype(int))
        rej = lambda e: float(1 / max(f[min(np.searchsorted(tp, e), len(f) - 1)], 1e-7))
        out['per_class'][CLASSES[c]] = dict(op_eff=float((pred[yy] == c).mean()), op_rej=float(1 / max((pred[~yy] == c).mean(), 1e-7)),
                                            auc=float(roc_auc_score(yy, P[:, c])), rej30=rej(.3), rej50=rej(.5), rej80=rej(.8),
                                            tpr=[round(float(x), 4) for x in tp[keep]], fpr=[round(float(x), 6) for x in f[keep]])
    return out
