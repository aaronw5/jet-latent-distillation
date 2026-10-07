"""The page of a direct-128 model (S15 family): the 128 class-token neurons each written as per-particle formulas
summed with ParT's attention weights, then ParT's last layer. Laid out like the JEDI-linear explorer: one drop-down
per neuron (what it measures, role, importance, every if-statement its own drop-down, the formula, complete Python,
particle groups, by class), the 16 heads that supply the weights, how the 128 neurons become the class scores,
a jet explorer, notation and definitions, the paper's metrics.

  python -m jetdistill.research.direct_page S15p OUT_DIR"""
import html, json, shutil, sys, pathlib
import numpy as np
from ..config import RESULTS, CLASSES
from ..pipeline import jets
from ..part.clsfit import PFEAT, particle_features
from ..part.network import ParTNetwork
from .nbr import NBR, nbr_features
from .heads import extract, phi_pooled, rows_of_split, OUT
from .one_term import extend
from .heads_page import CSS, TYPES, svg_profile
from .heads_defs import DEFS, WORDS, PHRASE, PK_DEF, term_math, group_words

NV = 38
INFO = {'S15p': ('The 128 neurons directly — per-particle formulas, ParT’s selection, pruned',
                 'The attention weights of the 16 heads (which particles each head looks at, per jet) are ParT’s, computed from its particle blocks; and ParT’s last layer (128 → 10).',
                 'Each of the 128 class-token neurons is a sum over the particles of formulas of each particle’s own physics, one formula per head, weighted by that head’s attention. Nothing else of ParT: no values, no out-projection, no MLPs, no LayerNorms. Pruned to the inputs each neuron needs and re-tuned toward ParT’s probabilities; at most one term per input per head and neuron.'),
        'S15o': ('The 128 neurons directly — per-particle formulas, ParT’s selection, one term per input', None, None)}
INFO['S15o'] = (INFO['S15o'][0], INFO['S15p'][1], INFO['S15p'][2].replace(' Pruned to the inputs each neuron needs and re-tuned toward ParT’s probabilities;', ' Re-tuned toward ParT’s probabilities;'))
INFO['S15q'] = ('The 128 neurons directly — per-particle formulas, ParT’s selection, statements pruned', INFO['S15p'][1], INFO['S15p'][2] + ' Then single statements removed, smallest first, within 0.1 pt, re-tuned.')
INFO['S17'] = ('The 128 neurons directly — one statement per input, shared by the heads', INFO['S15p'][1], INFO['S15p'][2] + ' Then every (neuron, input) keeps one statement — one kind and threshold shared by all 16 heads, each head with its own coefficient — re-tuned.')
INFO['S18q'] = ('The 10 class scores directly — per-particle formulas, ParT’s selection, pruned', INFO['S15p'][1].replace(' and ParT’s last layer (128 → 10)', ''), 'Each of the 10 class scores (logits) is a sum over the particles of formulas of each particle’s own physics, one formula per head, weighted by that head’s attention. Nothing else of ParT: no values, no MLPs, no LayerNorms, not even its last layer. At most one term per input per head and class, pruned to the inputs each class needs, single statements pruned, re-tuned toward ParT’s probabilities.')
INFO['C1q'] = ('Formula weights and formula neurons — only ParT’s last layer kept', 'ParT’s last layer (128 → 10). Nothing else: the attention weights come from the selection formulas (W1q, tuned with the neurons).', 'Which particles each head looks at: per head a score formula of each particle’s physics (softmax over the jet) and a jet-level formula for the class token’s share. What is read off them: the 128 neurons as per-particle formulas summed with those weights. At most one term per input, pruned, re-tuned toward ParT’s probabilities.')
TYPE5 = ['charged hadron', 'neutral hadron', 'photon', 'electron', 'muon']
HN = lambda h: f'b{h // 8 + 1}h{h % 8 + 1}'


def load_model(tag):
    P = dict(np.load(OUT / f'{tag}_model.npz')); K = 11 * NV + 2; return dict(W=P['W'].reshape(16, K, -1).astype(np.float64), knv=P['knv'], kn=P['kn'], names=(PFEAT + NBR)[:NV], logits=bool(P.get('logits', False)), alpha_from=str(P.get('alpha_from', '')))


def cols(f): return [f] + [NV + k * NV + f for k in range(5)] + [6 * NV + k * NV + f for k in range(5)]


def term_of(W, h, f, n, knv):
    """the (coef, kind, threshold) terms of input f in head h of neuron n (at most one in the one-term models)"""
    out = []
    if W[h, f, n]: out.append((float(W[h, f, n]), 'lin', None))
    for k in range(5):
        if W[h, NV + k * NV + f, n]: out.append((float(W[h, NV + k * NV + f, n]), 'gt', float(knv[k, f])))
        if W[h, 6 * NV + k * NV + f, n]: out.append((float(W[h, 6 * NV + k * NV + f, n]), 'lt', float(knv[k, f])))
    return out


def analyze(tag, n_dev=20000, n_ex=6, device='mps', log=print):
    import torch
    from sklearn.metrics import roc_auc_score
    m = load_model(tag); W, knv, names = m['W'], m['knv'], m['names']; model = ParTNetwork('full').model; model.eval()
    J = jets('full', 'dev'); rows = rows_of_split('dev', n_dev); _, A, M, L, _ = extract(model, 'dev', n_dev, device); ref = L.argmax(1); ytrue = np.asarray(J['y'][rows])
    if m['alpha_from']:                                                                       # combined models: the weights from the selection formulas
        from .direct_alpha import zfun, formula_alpha
        A = formula_alpha(m['alpha_from'], J, rows, model, zfun(model, device)(J, rows, M), device); log(f'  weights from the selection formulas {m["alpha_from"]}')
    Pd, _ = phi_pooled(J, rows, A, m['kn']); E = extend(Pd, knv, NV); V = np.einsum('nhk,hko->no', E, W)                           # the 128 neurons of the formula model
    NO = W.shape[2]; LG = m['logits']
    if LG: Hp = L.astype(np.float32); FW = np.eye(10); FB = np.zeros(10)                                     # S18: the outputs ARE the class scores
    else: Hp = np.asarray(J['H'][rows], np.float32); FW = model.fc[0].weight.detach().cpu().numpy().astype(np.float64); FB = model.fc[0].bias.detach().cpu().numpy().astype(np.float64)
    Lm = V @ FW.T + FB; pred = Lm.argmax(1); a0 = float((pred == ref).mean()); log(f'  {tag}: {100 * a0:.2f}% on {n_dev} dev jets')
    F0, ok = particle_features(J, rows, ctx=[]); Fn = nbr_features(J['x'][rows], J['ext'][rows], J['jet'][rows]); X = np.concatenate([F0, Fn], -1)[..., :NV]
    d0s = np.abs(np.asarray(J['ext'][rows][..., 3], np.float32)) / np.maximum(np.asarray(J['ext'][rows][..., 4], np.float32), 1e-6); npart = ok.sum(1); nmean = npart.mean()
    XR = X[ok].astype(np.float32); TR = np.concatenate([XR] + [np.maximum(0, XR - knv[k]) for k in range(5)] + [np.maximum(0, knv[k] - XR) for k in range(5)], -1)
    AR = np.stack([A[h // 8][:, h % 8, 1:][ok] for h in range(16)]); YR = np.repeat(ytrue, npart); JR = np.repeat(np.arange(n_dev), npart)
    tp = F0[..., 8:13][ok]; KR = {'charged hadron': tp[:, 0], 'neutral hadron': tp[:, 1], 'photon': tp[:, 2], 'lepton': tp[:, 3] + tp[:, 4]}; ZR = np.exp(F0[..., 2][ok]) > .05
    HIST = {}
    for f in range(NV):
        lo_, hi_ = np.quantile(XR[:, f], [.01, .99]); hi_ = hi_ if hi_ > lo_ else lo_ + 1; cnts, edges = np.histogram(XR[:, f], 40, (lo_, hi_)); HIST[names[f]] = dict(edges=[round(float(e), 5) for e in edges], counts=[int(c) for c in cnts])
    # per particle: f_h(x_i) for every head and neuron (sparse rows only), and its α-weighted total contribution g_n(i)
    FH = np.zeros((16, len(XR), NO), np.float32); G = np.zeros((len(XR), NO), np.float32)
    for h in range(16):
        nz = np.flatnonzero(np.abs(W[h, :11 * NV]).sum(1))
        if len(nz): FH[h] = TR[:, nz] @ W[h, nz].astype(np.float32)
        G += AR[h][:, None] * FH[h]
    ncls = np.array([(ytrue == c).sum() for c in range(10)]).clip(1)
    byclass = lambda con: [float(con[YR == c].sum() / ncls[c]) for c in range(10)]
    def direction(terms_all, f):
        lo, hi = float(knv[0, f]), float(knv[-1, f]); xs = np.linspace(lo - .3 * (hi - lo + 1e-9), hi + .3 * (hi - lo + 1e-9), 25)
        g = sum(c * (xs if k == 'lin' else np.maximum(0, xs - t) if k == 'gt' else np.maximum(0, t - xs)) for c, k, t in terms_all); return 1 if np.polyfit(xs, g, 1)[0] >= 0 else -1
    neurons = []; sdV = V.std(0)
    for n in range(NO):
        ins = []; FIRE = {}
        for f in range(NV):
            heads_f = []
            for h in range(16):
                for c, k, t in term_of(W, h, f, n, knv):
                    xf = XR[:, f]; fire = np.ones(len(xf), bool) if k == 'lin' else (xf > t) if k == 'gt' else (xf < t)
                    con = AR[h] * c * (xf if k == 'lin' else np.maximum(0, xf - t) if k == 'gt' else np.maximum(0, t - xf))
                    heads_f.append(dict(head=h, coef=c, kind=k, thr=t, fires=float(fire.mean()), kinds={k2: float(v2[fire].mean()) if fire.any() else 0.0 for k2, v2 in KR.items()},
                                        hard=float(ZR[fire].mean()) if fire.any() else 0.0, by_class=byclass(con), size=float(np.abs(con).sum() / n_dev)))
            if not heads_f: continue
            imp = float(sum(E[:, h, cols(f)] @ W[h, cols(f), n] for h in range(16)).std()); heads_f.sort(key=lambda d: -d['size'])
            shared = len({(hd['kind'], hd['thr']) for hd in heads_f}) == 1; h0 = heads_f[0]; xf = XR[:, f]
            fire = np.ones(len(xf), bool) if h0['kind'] == 'lin' else (xf > h0['thr']) if h0['kind'] == 'gt' else (xf < h0['thr']); FIRE[f] = fire
            stat = dict(kind=h0['kind'], thr=h0['thr'], fires=h0['fires'], kinds=h0['kinds'], hard=h0['hard'], size=float(sum(hd['size'] for hd in heads_f)),
                        by_class=[float(sum(hd['by_class'][c] for hd in heads_f)) for c in range(10)], coefs=[(hd['head'], hd['coef']) for hd in heads_f])
            ins.append(dict(feature=names[f], f=f, importance=imp, heads=heads_f, shared=shared, stat=stat, direction=direction([(d['coef'], d['kind'], d['thr']) for d in heads_f], f)))
        # like terms on categorical inputs: the one-hot particle type (5 inputs) and the charge (−1, 0, +1) → one lookup statement each (exact)
        piece_at = lambda x, c, k, t: c * (x if k == 'lin' else max(0.0, x - t) if k == 'gt' else max(0.0, t - x))
        for cat, members, levels, colidx in (('particle type', TYPE5, TYPE5, {nm: 8 + i for i, nm in enumerate(TYPE5)}), ('charge', ['charge'], [-1, 0, 1], {'charge': 7})):
            mem = [d for d in ins if d['feature'] in members]
            if len(mem) < (2 if cat == 'particle type' else 1): continue
            table = {}                                                                                   # head → {level: value}
            for d in mem:
                for hd in d['heads']:
                    for lv in levels:
                        x = (1.0 if lv == d['feature'] else 0.0) if cat == 'particle type' else float(lv)
                        table.setdefault(hd['head'], {}).setdefault(lv, 0.0); table[hd['head']][lv] += piece_at(x, hd['coef'], hd['kind'], hd['thr'])
            lvl_of = np.argmax(tp, 1) if cat == 'particle type' else np.rint(XR[:, 7]).astype(int)       # the level of every real particle
            con = np.zeros(len(XR), np.float32)
            for h, tb in table.items():
                vals = np.array([tb[lv] for lv in levels], np.float32); con += AR[h] * vals[lvl_of if cat == 'particle type' else lvl_of + 1]
            byc = byclass(con); imp = float(sum(d['importance'] for d in mem))
            share = {str(lv): float((lvl_of == (i if cat == 'particle type' else lv)).mean()) for i, lv in enumerate(levels)}
            ins = [d for d in ins if d not in mem] + [dict(feature=cat, f=-1, importance=imp, heads=[], shared=True, categorical=True, levels=[str(lv) for lv in levels], level_share=share,
                                                            table={HN(h): {str(lv): float(tb[lv]) for lv in levels} for h, tb in sorted(table.items())}, by_class=byc, direction=1,
                                                            stat=dict(kind='cat', thr=None, fires=1.0, kinds={}, hard=0.0, size=float(np.abs(con).sum() / n_dev), by_class=byc, coefs=[]))]
        ins.sort(key=lambda d: -d['importance']); v = V[:, n]
        Va = Lm - np.outer(v - v.mean(), FW[:, n]); drop = a0 - float((Va.argmax(1) == ref).mean())
        means = [float(v[ytrue == c].mean()) for c in range(10)]; auc = [float(roc_auc_score(ytrue == c, v)) if 0 < (ytrue == c).sum() < len(v) else .5 for c in range(10)]
        eff = [float(FW[c, n] * sdV[n]) for c in range(10)]; strength = float(np.abs(FW[:, n]).mean() * sdV[n]); r2 = float(1 - ((v - Hp[:, n]) ** 2).mean() / (Hp[:, n].var() + 1e-9))
        order = np.argsort(-np.abs(eff))[:3]
        role = '; '.join(f'{"raises" if eff[c] > 0 else "lowers"} the {CLASSES[c]} score ({eff[c]:+.2f} per standard deviation of the neuron)' for c in order if abs(eff[c]) > .02) or 'hardly enters any class score'
        tot = sum(d['importance'] for d in ins) or 1.0; parts = []
        for d in ins[:3]:
            hi_p, lo_p = ('particles by type', 'particles by type') if d.get('categorical') else PHRASE.get(d['feature'], (f'high {d["feature"]}', f'low {d["feature"]}')); parts.append(f'{"rises" if d["direction"] > 0 else "falls"} with {d["feature"]} — up for {hi_p if d["direction"] > 0 else lo_p} ({100 * d["importance"] / tot:.0f} %)')
        ph = lambda d: ('particles by type' if d['feature'] == 'particle type' else 'particles by charge') if d.get('categorical') else PHRASE.get(d['feature'], ('high ' + d['feature'], 'low ' + d['feature']))[0 if d['direction'] > 0 else 1]
        title = (ph(ins[0]) if ins else 'constant') + (f', {ph(ins[1])}' if len(ins) > 1 else '')
        big = int(np.argmax(means)); sep = int(np.argmax(np.abs(np.array(auc) - .5)))
        hs = np.array([float(np.abs(E[:, h, :] @ W[h, :, n] - (E[:, h, :] @ W[h, :, n]).mean()).mean()) for h in range(16)]); hs = hs / (hs.sum() or 1)
        groups = []; hinges = sorted([d for d in ins if d['stat']['kind'] not in ('lin', 'cat')], key=lambda d: -d['importance'] * min(d['stat']['fires'], 1 - d['stat']['fires']))[:8]   # the 8 most discriminating if-statements (importance × balance of firing) → ≤ 256 patterns
        if hinges:
            awt = AR.mean(0); contrib = G[:, n]; tot_c = np.abs(contrib).sum() or 1.0; fbar = (AR * FH[:, :, n]).sum(0) / np.maximum(AR.sum(0), 1e-12)
            Mk = np.stack([FIRE[d['f']] for d in hinges], 1); code = (Mk * (1 << np.arange(len(hinges)))).sum(1); short = lambda d: f"{d['feature']} {'>' if d['stat']['kind'] == 'gt' else '<'} {d['stat']['thr']:.3g}"
            pats = []
            for cval in np.unique(code):
                g = code == cval; on = [short(d) for b, d in enumerate(hinges) if cval >> b & 1]; off = [short(d) for b, d in enumerate(hinges) if not cval >> b & 1]
                pats.append(dict(name=('fires: ' + ', '.join(on) if on else 'none of the statements fires') + (' · not: ' + ', '.join(off) if off and on and len(off) <= 3 else ''), fires=on, share=float(g.mean()), mean_f=float(fbar[g].mean()), mean_w=float(awt[g].mean() * nmean),
                                 contrib=float(contrib[g].sum() / tot_c), abs_share=float(np.abs(contrib[g]).sum() / tot_c), kinds={k_: float(v_[g].mean()) for k_, v_ in KR.items()}, hard=float(ZR[g].mean()), by_class=[float(contrib[g & (YR == c)].sum() / ncls[c]) for c in range(10)]))
            pats.sort(key=lambda g: -g['abs_share']); groups = pats[:10]
        neurons.append(dict(n=n, vmean=float(v.mean()), vsd=float(v.std()), title=title, inputs=ins, strength=strength, drop=drop, r2=r2, means=means, auc=auc, effect=eff, fc=[float(FW[c, n]) for c in range(10)], bias=float(W[:, -1, n].sum()),
                            cls=[float(W[h, -2, n]) for h in range(16)], head_share=hs.round(4).tolist(), groups=groups,
                            measures=('Sums over the particles of the jet, with each head’s weights, a formula that ' + '; '.join(parts) + '.') if ins else 'A constant plus the class-token terms: no particle input.',
                            role=f'This neuron {role}; without it (held at its mean), agreement with ParT changes by {-100 * drop:+.2f} pt. It reproduces ParT’s neuron {n + 1} with R² = {r2:.2f}.',
                            scale=f'largest for {CLASSES[big]} jets; separates {CLASSES[sep]} jets best (AUC {max(auc[sep], 1 - auc[sep]):.2f})'))
        if n % 32 == 31: log(f'  neurons 1–{n + 1} analysed')
    st = np.array([d['strength'] for d in neurons]); q80, q50 = np.quantile(st, .8), np.quantile(st, .5)
    classes = []
    if not LG:
        Cf = np.zeros((n_dev, 10, NV), np.float32)                                                            # per jet: what each input adds to each class score (through all neurons)
        for f in range(NV): Cf[:, :, f] = np.einsum('nhk,hko->no', E[:, :, cols(f)], W[:, cols(f), :]) @ FW.T
        for c in range(10):
            sp = np.abs(FW[c]) * sdV; tot = sp.sum() or 1.0; order = np.argsort(-sp); keep = []; acc = 0.0
            for n in order:
                if acc >= .95 * tot or sp[n] < 1e-6: break
                acc += sp[n]; keep.append(dict(n=int(n), w=float(FW[c, n]), spread=float(sp[n]), share=float(sp[n] / tot), title=neurons[n]['title'], sign='raises' if FW[c, n] > 0 else 'lowers'))
            fi = Cf[:, c].std(0); fo = np.argsort(-fi)[:10]; sc = Lm[:, c]
            classes.append(dict(c=c, bias=float(FB[c]), neurons=keep, inputs=[dict(feature=names[f], spread=float(fi[f]), sign=float(np.corrcoef(Cf[:, c, f], sc)[0, 1]) if Cf[:, c, f].std() > 0 else 0.0) for f in fo],
                                means=[float(sc[ytrue == k].mean()) for k in range(10)], auc=float(roc_auc_score(ytrue == c, sc)) if 0 < (ytrue == c).sum() < n_dev else .5,
                                agree=float((pred[ref == c] == c).mean()) if (ref == c).any() else 0.0, n_const=int(sum(1 for n in range(NO) if not neurons[n]['inputs'] and abs(FW[c, n]) > 0))))
    for d in neurons: d['importance'] = 'major' if d['strength'] >= q80 else 'moderate' if d['strength'] >= q50 else 'minor'
    # the heads: what they select (weight profiles), how much of the model goes through each, which neurons they serve most
    heads = []
    for h in range(16):
        b = h // 8; al = A[b, :, h % 8, 1:]; rel = al * npart[:, None]; sel = {}
        for name, v, bins in (('ln pT/pT_jet', F0[..., 2], np.linspace(-7, -0.5, 9)), ('ΔR', F0[..., 4], np.linspace(0, 0.8, 9)), ('|d0|/σ', d0s, np.array([0, .5, 1, 2, 3, 5, 10, 1e9]))):
            idx = np.clip(np.searchsorted(bins, v) - 1, 0, len(bins) - 2); sel[name] = dict(edges=[float(e) if e < 1e8 else None for e in bins], weight=[float(rel[ok & (idx == i)].mean()) if (ok & (idx == i)).any() else None for i in range(len(bins) - 1)])
        for name, msk in (('charged', F0[..., 7] != 0), ('photon', F0[..., 10] > 0), ('lepton', (F0[..., 11] + F0[..., 12]) > 0)): sel[name] = float(rel[ok & msk].mean()) if (ok & msk).any() else None
        out = []; w = [v for v in sel['ln pT/pT_jet']['weight'] if v is not None]
        if len(w) > 2 and w[-1] > 1.6 * max(w[0], 1e-9): out.append('the hard particles')
        w = [v for v in sel['ΔR']['weight'] if v is not None]
        if len(w) > 2 and w[0] > 1.4 * max(w[-1], 1e-9): out.append('particles near the axis')
        elif len(w) > 2 and w[-1] > 1.4 * max(w[0], 1e-9): out.append('wide-angle particles')
        for k, nm, th in (('lepton', 'leptons', 1.5), ('charged', 'charged particles', 1.15), ('photon', 'photons', 1.2)):
            if sel.get(k) and sel[k] > th: out.append(nm)
        w = [v for v in sel['|d0|/σ']['weight'] if v is not None]
        if len(w) > 2 and w[-1] > 1.5 * max(w[0], 1e-9): out.append('displaced tracks')
        share = np.array([d['head_share'][h] * d['strength'] for d in neurons]); top = np.argsort(-share)[:5]
        Vh = V - np.einsum('nk,ko->no', E[:, h, :], W[h]) + np.einsum('nk,ko->no', E[:, h, :], W[h]).mean(0); drop = a0 - float(((Vh @ FW.T + FB).argmax(1) == ref).mean())
        nin = int(np.any(np.abs(W[h, :11 * NV]) > 0, axis=1).sum()); nt = int((W[h, :11 * NV] != 0).sum())
        heads.append(dict(block=b + 1, head=h % 8 + 1, selects=', '.join(out[:3]) if out else 'all particles about equally', self_weight=float(A[b, :, h % 8, 0].mean()), eff_particles=float(np.exp(-(al * np.log(al + 1e-12)).sum(1)).mean()),
                          selection=sel, drop=drop, inputs=nin, terms=nt, serves=[(int(n), float(share[n] / (share.sum() or 1))) for n in top],
                          by_class=[dict(cls=CLASSES[c], eff=float(np.exp(-(al * np.log(al + 1e-12)).sum(1))[ytrue == c].mean()), self=float(A[b, :, h % 8, 0][ytrue == c].mean())) for c in range(10)]))
    def stmts_of(d):
        out = []
        for din in d['inputs']:
            if din.get('categorical'): out.append(dict(t='cat', f=din['feature'], label='value by ' + din['feature'], table=din['table']))
            elif din.get('shared'): out.append(dict(t='sh', f=din['feature'], fi=din['f'], kind=din['stat']['kind'], thr=din['stat']['thr'], coefs=din['stat']['coefs'], label=stmt(dict(coef=1.0, kind=din['stat']['kind'], thr=din['stat']['thr']), din['feature']).replace('add +1 · ', 'add c_h · ')))
            else: out += [dict(t='h', f=din['feature'], fi=din['f'], kind=hd['kind'], thr=hd['thr'], coefs=[(hd['head'], hd['coef'])], label=f'[{HN(hd["head"])}] ' + stmt(hd, din['feature'])) for hd in din['heads']]
        return out
    def amounts(st_list, sl, tpj, qj):
        res = []
        for st in st_list:
            if st['t'] == 'cat':
                lvl = [TYPE5[k] for k in np.argmax(tpj, 1)] if st['f'] == 'particle type' else [str(int(round(q))) for q in qj]
                amt = sum(float(AR[int(hn[1]) * 8 - 8 + int(hn[3]) - 1, sl] @ np.array([tb[l] for l in lvl])) for hn, tb in st['table'].items()); nf = len(lvl)
            else:
                x = XR[sl, st['fi']]; k, t = st['kind'], st['thr']; pc = x if k == 'lin' else np.maximum(0, x - t) if k == 'gt' else np.maximum(0, t - x)
                amt = float(sum(c * (AR[h, sl] @ pc) for h, c in st['coefs'])); nf = int(len(x) if k == 'lin' else ((x > t) if k == 'gt' else (x < t)).sum())
            res.append([round(amt, 4), nf])
        return res
    for d in neurons:
        if d['inputs']: d['stmts'] = [dict(label=st['label'], f=st['f']) for st in stmts_of(d)]
    # example jets: 6 per ParT class; every head's weights, each particle's inputs and its contribution to every neuron
    rng = np.random.default_rng(0); ex = []; start = np.concatenate([[0], np.cumsum(npart)[:-1]])
    for c in range(10):
        cand = np.flatnonzero((ref == c) & (npart >= 8)); pick = list(cand[np.argsort(npart[cand])[:n_ex // 2]]) + list(rng.choice(cand, n_ex - n_ex // 2, replace=False))
        for i in pick:
            ps = np.flatnonzero(ok[i]); sl = slice(start[i], start[i] + npart[i])
            parts = [dict(eta=round(float(F0[i, p, 5]), 4), phi=round(float(F0[i, p, 6]), 4), z=float(np.exp(F0[i, p, 2])), dr=float(F0[i, p, 4]), type=TYPES[int(np.argmax(F0[i, p, 8:13]))], q=int(F0[i, p, 7]), d0s=float(d0s[i, p])) for p in ps]
            sm = np.exp(Lm[i] - Lm[i].max()); sm /= sm.sum(); sp = np.exp(L[i] - L[i].max()); sp /= sp.sum()
            ex.append(dict(part=int(ref[i]), truth=int(ytrue[i]), model=int(pred[i]), p_model=sm.round(3).tolist(), p_part=sp.round(3).tolist(), n=int(npart[i]), particles=parts,
                           A=np.round(AR[:, sl], 4).tolist(), self=[round(float(A[h // 8, i, h % 8, 0]), 4) for h in range(16)], X=np.round(XR[sl], 4).tolist(), C=np.round(G[sl], 4).tolist(),
                           V=np.round(V[i], 3).tolist(), H=np.round(Hp[i], 3).tolist(), logit=np.round(Lm[i], 3).tolist(), logit_part=np.round(L[i], 3).tolist(),
                           ST={d['n']: amounts(stmts_of(d), sl, tp[sl], XR[sl, 7]) for d in neurons if d['inputs']},
                           HC={d['n']: [[round(float(AR[h, sl] @ FH[h, sl, d['n']]), 4), round(float(W[h, -2, d['n']] * A[h // 8, i, h % 8, 0]), 4)] for h in range(16)] for d in neurons if d['inputs']},
                           B={d['n']: round(float(W[:, -1, d['n']].sum()), 4) for d in neurons if d['inputs']},
                           FH={d['n']: np.round(FH[:, sl, d['n']], 3).tolist() for d in neurons if d['inputs']}))
    return dict(tag=tag, agreement=a0, n_dev=n_dev, neurons=neurons, heads=heads, classes=classes, logits=LG, examples=ex, hists=HIST, fc=dict(W=FW.round(5).tolist(), b=FB.round(5).tolist()),
                pairs=int(sum(len(d['inputs']) for d in neurons)), terms=int((W[:, :11 * NV] != 0).sum()), r2_mean=float(np.mean([d['r2'] for d in neurons])))


def stmt(hd, feature):
    v = f'particle["{feature}"]'; c, k, t = hd['coef'], hd['kind'], hd['thr']
    return f'add {c:+.4g} · {v}' if k == 'lin' else f'if {v} > {t:.4g}:  add {c:+.4g} · ({v} − {t:.4g})' if k == 'gt' else f'if {v} < {t:.4g}:  add {c:+.4g} · ({t:.4g} − {v})'


def neuron_python(d):
    L = [f'# particle["name"] = that particle’s input called “name” (see the table of inputs at the bottom of the page)',
         f'def neuron_{d["n"] + 1}(particles, alpha, alpha_cls):',
         f'    """neuron {d["n"] + 1} of the 128: for each of the 16 heads h (0–7 = block 1 heads 1–8, 8–15 = block 2 heads 1–8), the sum over the',
         f'    particles of alpha[h][i] * f_h(particle i), plus c_h * alpha_cls[h]; then the bias. alpha[h]: head h\'s weights of the particles (ParT\'s)."""',
         f'    total = {d["bias"]:.6g}']
    for h in range(16):
        lines = [(hd, din['feature']) for din in d['inputs'] for hd in din['heads'] if hd['head'] == h]; cats = [din for din in d['inputs'] if din.get('categorical') and HN(h) in din['table']]
        if not lines and not cats and abs(d['cls'][h]) < 1e-12: continue
        L.append(f'    # head {HN(h)}')
        if lines or cats:
            L += [f'    for particle, a in zip(particles, alpha[{h}]):', '        f = 0.0']
            for din in cats:
                key = 'particle["type"]' if din['feature'] == 'particle type' else 'int(particle["charge"])'; tb = din['table'][HN(h)]
                L.append(f'        f += {{{", ".join((repr(lv) if din["feature"] == "particle type" else lv) + f": {val:.6g}" for lv, val in tb.items())}}}[{key}]')
            for hd, feat in lines:
                v = f'particle["{feat}"]'; c, k, t = hd['coef'], hd['kind'], hd['thr']
                L.append(f'        f += {c:.6g} * {v}' if k == 'lin' else f'        if {v} > {t:.6g}: f += {c:.6g} * ({v} - {t:.6g})' if k == 'gt' else f'        if {v} < {t:.6g}: f += {c:.6g} * ({t:.6g} - {v})')
            L.append('        total += a * f')
        L.append(f'    total += {d["cls"][h]:.6g} * alpha_cls[{h}]')
    L.append('    return total'); return '\n'.join(L)


def neuron_formula(d):
    out = [f'# neuron {d["n"] + 1} = b + Σ_h [ Σ_i α_hi · f_h(particle i) + c_h · α_h,cls ],  h over the 16 heads, i over the jet’s particles',
           f'# particle["name"] = that particle’s input called “name” (table of inputs at the bottom); ±: how much the input moves the neuron across jets', f'b = {d["bias"]:+.4g}']
    for h in range(16):
        terms = [(hd, din) for din in d['inputs'] for hd in din['heads'] if hd['head'] == h]; cats = [din for din in d['inputs'] if din.get('categorical') and HN(h) in din['table']]
        if not terms and not cats: continue
        catx = ' '.join('+ value[' + ('particle["type"]' if din['feature'] == 'particle type' else 'particle["charge"]') + ']{' + ', '.join(f'{lv}: {val:+.3g}' for lv, val in din['table'][HN(h)].items()) + '}' for din in cats)
        out.append(f'f_{HN(h)}(particle) = ' + ' '.join(term_math(hd['coef'], hd['kind'], hd['thr'], din['feature']) for hd, din in terms) + (' ' + catx if catx else '') + f'      # c = {d["cls"][h]:+.3g}')
    return '\n'.join(out)


def python_export(tag, an):
    return f'''"""Model {tag}: ParT's 128 class-token neurons written directly as per-particle formulas with ParT's attention weights.
neuron_n = b_n + sum_h [ sum_i alpha[h, i] * f_hn(x_i) + c_hn * alpha_cls[h] ];  class scores = FC_W @ neurons + FC_B (ParT's last layer).
x_i: the {NV} per-particle inputs of particle i (jetdistill/part/clsfit.py particle_features + research/nbr.py nbr_features, first {NV} columns);
alpha[h]: head h's attention weights of the particles (h = 0..7 block 1, 8..15 block 2), alpha_cls[h] its weight on the class token (ParT's own).
f_hn(x) = sum over terms w * t(x), t(x) in {{x_f, max(0, x_f - KN[k, f]), max(0, KN[k, f] - x_f)}}; the coefficients are in {tag}_model.npz."""
import numpy as np
P = np.load('{tag}_model.npz'); W = P['W'].reshape(16, {11 * NV + 2}, 128); KN = P['knv']          # W[h]: rows = terms [x (38), x > KN_k (5 x 38), x < KN_k (5 x 38), alpha_cls, 1]
FC_W = np.array({json.dumps(an['fc']['W'])})
FC_B = np.array({json.dumps(an['fc']['b'])})
NAMES = {json.dumps((PFEAT + NBR)[:NV], ensure_ascii=False)}


def terms(x):                                                   # x: (n, {NV}) -> (n, {11 * NV})
    return np.concatenate([x] + [np.maximum(0, x - KN[k]) for k in range(5)] + [np.maximum(0, KN[k] - x) for k in range(5)], -1)


def neurons(X, alpha, alpha_cls):
    """X: (n, {NV}) per-particle inputs of one jet; alpha: (16, n); alpha_cls: (16,) -> the 128 neurons"""
    T = terms(X); out = W[:, -1].sum(0).copy()
    for h in range(16): out += alpha[h] @ (T @ W[h, :-2]) + alpha_cls[h] * W[h, -2]
    return out


def class_scores(X, alpha, alpha_cls): return FC_W @ neurons(X, alpha, alpha_cls) + FC_B
'''


def page(tag, outdir, device='mps', log=print):
    an = analyze(tag, device=device, log=log); title, kept, formula = INFO[tag]; outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True); pct = lambda x: f'{100 * x:.2f}%'
    te = next((json.loads(p.read_text()) for p in (OUT / f'{tag}_metrics_full_test.json', OUT / f'{tag}_model_metrics_full_test.json') if p.exists()), None)
    base = json.loads((RESULTS / 'all_plus' / 'full' / 'metrics.json').read_text()); bm = next(x for x in base['models'] if x.get('terms') == 1083)
    if te:
        cl = list(te['rej']); t = '<table><tr><th>model</th><th class="num">same class as ParT</th><th class="num">accuracy</th><th class="num">AUC</th>' + ''.join(f'<th class="num">Rej<sub>{int(te["rej"][c]["eff"] * 1000) / 10:g}%</sub> {c}</th>' for c in cl) + '</tr>'
        for name, mm, ag in (('ParT', te['part'], None), ('jet-level formula, all_plus (1083 terms)', bm['paper'], bm['same_as_network']), (title, te, te['agreement'])):
            t += f'<tr><td>{html.escape(name)}</td><td class="num">{"—" if ag is None else pct(ag)}</td><td class="num">{mm["accuracy"]:.4f}</td><td class="num">{mm["auc"]:.4f}</td>' + ''.join(f'<td class="num">{mm["rej"][c]["rej"]:.0f}</td>' for c in cl) + '</tr>'
        pc = '<table><tr><th>class</th><th class="num">same class as ParT</th><th class="num">accuracy</th></tr>' + ''.join(f'<tr><td>{CLASSES[p["c"]]}</td><td class="num">{pct(p["agreement"])}</td><td class="num">{pct(p["accuracy"])}</td></tr>' for p in te['per_class']) + '</table>'
        metrics_html = f'<h2>The ParT paper’s metrics (2,000,000 test jets)</h2><div class="card">{t}</table></div><h2>Per class</h2><div class="card">{pc}</div>'
        kpi = f'<span class="kpi">same class as ParT (test)<b>{pct(te["agreement"])}</b></span><span class="kpi">accuracy<b>{te["accuracy"]:.4f}</b><span class="cnt">ParT {te["part"]["accuracy"]:.4f}</span></span><span class="kpi">AUC<b>{te["auc"]:.4f}</b><span class="cnt">ParT {te["part"]["auc"]:.4f}</span></span>'
    else:
        metrics_html = ''; kpi = f'<span class="kpi">same class as ParT<b>{pct(an["agreement"])}</b><span class="cnt">balanced validation sample, {an["n_dev"]:,} jets (test pending)</span></span>'
    act = [d for d in an['neurons'] if d['inputs']]; LG = an.get('logits', False); NL = (lambda n: f'{CLASSES[n]} score') if LG else (lambda n: f'Neuron {n + 1}')
    kpi += f'<span class="kpi">{"class scores" if LG else "neurons"} with particle inputs<b>{len(act)} of {len(an["neurons"])}</b><span class="cnt">the other {len(an["neurons"]) - len(act)} are constants (ParT’s last layer barely reads them)</span></span><span class="kpi">(neuron, input) pairs<b>{an["pairs"]}</b><span class="cnt">{an["terms"]} statements over the 16 heads</span></span><span class="kpi">mean R² to ParT’s neurons<b>{an["r2_mean"]:.2f}</b></span>'
    imp_col = {'major': '#1d2433', 'moderate': '#98a2b3', 'minor': '#e4e7ec'}
    def bars(vals, lab, fmt, center=None):
        mx = max(abs(x - (center or 0)) for x in vals) or 1
        return '<table class="bars">' + ''.join(f'<tr><td>{CLASSES[c]}</td><td style="width:150px"><div class="bar" style="width:{130 * abs(v - (center or 0)) / mx:.0f}px;background:{"#2f855a" if v - (center or 0) >= 0 else "#c05621"}"></div></td><td class="num">{fmt(v)}</td></tr>' for c, v in enumerate(vals)) + f'</table><div class="cnt">{lab}</div>'
    def terms_html(d):
        out = []
        for din in d['inputs']:
            feat = din['feature']
            if din.get('categorical'):
                st = din['stat']; byc = sorted(enumerate(st['by_class']), key=lambda q: -abs(q[1]))[:4]; mxc = max(abs(v2) for _, v2 in byc) or 1
                cls = ''.join(f'<tr><td>{CLASSES[ci]}</td><td style="width:120px"><div class="bar" style="width:{100 * abs(v2) / mxc:.0f}px;background:{"#2f855a" if v2 >= 0 else "#c05621"}"></div></td><td class="num">{v2:+.3g}</td></tr>' for ci, v2 in byc)
                tot_h = {lv: sum(tb[lv] for tb in din['table'].values()) for lv in din['levels']}; key = 'particle["type"]' if feat == 'particle type' else 'particle["charge"]'
                line = f'add value by {feat}:  ' + ', '.join(f'{lv} {tot_h[lv]:+.3g}' for lv in din['levels']) + '   (summed over the heads; per head below)'
                rows = ''.join(f'<tr><td>{hn}</td>' + ''.join(f'<td class="num">{tb[lv]:+.3g}</td>' for lv in din['levels']) + '</tr>' for hn, tb in din['table'].items())
                out.append(f'''<details class="xterm"><summary><code>{html.escape(line)}</code> <span class="cnt">typical contribution ±{din["importance"]:.3g}; {len(din["table"])} head{"s" if len(din["table"]) != 1 else ""}</span></summary>
<div class="grid" style="margin:6px 0 4px 10px"><div><div><b>{html.escape(feat)}</b>: {"what kind of particle i is (exactly one of the five)" if feat == "particle type" else "its electric charge, −1, 0 or +1"}. The inputs {html.escape(", ".join(TYPE5) if feat == "particle type" else "charge")} are {"one-hot" if feat == "particle type" else "three-valued"}, so every if-statement on them is a fixed amount per kind: combined here into one lookup (exact).</div>
<div style="margin-top:4px">Applied to every particle i, weighted by each head’s attention for it: Σ<sub>h</sub> α<sub>hi</sub> · value<sub>h</sub>[{key}]. Share of particles: {html.escape(", ".join(f"{lv} {100 * sh:.0f} %" for lv, sh in din["level_share"].items()))}.</div></div>
<div><div class="cnt">value per head and kind</div><table><tr><th>head</th>{"".join(f"<th class=num>{html.escape(lv)}</th>" for lv in din["levels"])}</tr>{rows}</table></div>
<div><div class="cnt">what this statement adds to the neuron, per jet, by true class (all heads, α-weighted)</div><table class="bars">{cls}</table></div></div></details>''')
                continue
            if din.get('shared'):
                st = din['stat']; k = st['kind']; line = stmt(dict(coef=1.0, kind=k, thr=st['thr']), feat).replace('add +1 · ', 'add c_h · ')
                kinds = ', '.join(f'{k2} {100 * v2:.0f} %' for k2, v2 in sorted(st['kinds'].items(), key=lambda q: -q[1]) if v2 > .05)
                byc = sorted(enumerate(st['by_class']), key=lambda q: -abs(q[1]))[:4]; mxc = max(abs(v2) for _, v2 in byc) or 1
                cls = ''.join(f'<tr><td>{CLASSES[ci]}</td><td style="width:120px"><div class="bar" style="width:{100 * abs(v2) / mxc:.0f}px;background:{"#2f855a" if v2 >= 0 else "#c05621"}"></div></td><td class="num">{v2:+.3g}</td></tr>' for ci, v2 in byc)
                when = 'always (a linear term)' if k == 'lin' else f'for {100 * st["fires"]:.0f} % of all particles'
                coefs = ', '.join(f'{HN(h)} {c:+.3g}' for h, c in st['coefs'])
                out.append(f'''<details class="xterm"><summary><code>{html.escape(line)}</code> <span class="cnt">typical contribution ±{din["importance"]:.3g}; {len(st["coefs"])} head{"s" if len(st["coefs"]) != 1 else ""}</span></summary>
<div class="grid" style="margin:6px 0 4px 10px"><div><div><b>{html.escape(feat)}</b>: {html.escape(WORDS.get(feat, ""))} <span class="cnt">({html.escape(DEFS.get(feat, ""))})</span></div>
<div style="margin-top:4px">Applied to every particle i; what it adds is weighted by each head’s attention for that particle, with the head’s own coefficient c<sub>h</sub>: Σ<sub>h</sub> c<sub>h</sub> α<sub>hi</sub> · (…). <b>c<sub>h</sub></b>: {html.escape(coefs)}. Fires {when}{"" if k == "lin" else f"; those are {kinds}; hard (pT share > 5 %) {100 * st['hard']:.0f} %"}. The amount added grows with the distance from the threshold.</div></div>
<div><div class="cnt">distribution of this input over the particles (threshold in orange)</div><div class="hh" data-f="{html.escape(feat)}" data-thr="{'' if st['thr'] is None else st['thr']}"></div></div>
<div><div class="cnt">what this statement adds to the neuron, per jet, by true class (all heads, α-weighted)</div><table class="bars">{cls}</table></div></div></details>''')
                continue
            out.append(f'<div style="margin:6px 0 2px"><b>{html.escape(feat)}</b> <span class="cnt">— {html.escape(WORDS.get(feat, ""))}; typical contribution ±{din["importance"]:.3g}; {len(din["heads"])} statement{"s" if len(din["heads"]) != 1 else ""} (one per head that uses it)</span></div>')
            for hd in din['heads']:
                line = stmt(hd, feat); k = hd['kind']
                kinds = ', '.join(f'{k2} {100 * v2:.0f} %' for k2, v2 in sorted(hd['kinds'].items(), key=lambda q: -q[1]) if v2 > .05)
                byc = sorted(enumerate(hd['by_class']), key=lambda q: -abs(q[1]))[:4]; mxc = max(abs(v2) for _, v2 in byc) or 1
                cls = ''.join(f'<tr><td>{CLASSES[ci]}</td><td style="width:120px"><div class="bar" style="width:{100 * abs(v2) / mxc:.0f}px;background:{"#2f855a" if v2 >= 0 else "#c05621"}"></div></td><td class="num">{v2:+.3g}</td></tr>' for ci, v2 in byc)
                when = 'always (a linear term)' if k == 'lin' else f'for {100 * hd["fires"]:.0f} % of all particles'
                out.append(f'''<details class="xterm"><summary><span class="pill" style="background:#3b5b8c">head {HN(hd["head"])}</span> <code>{html.escape(line)}</code> <span class="cnt">size {hd["size"]:.3g} per jet</span></summary>
<div class="grid" style="margin:6px 0 4px 10px"><div><div><b>{html.escape(feat)}</b>: {html.escape(WORDS.get(feat, ""))} <span class="cnt">({html.escape(DEFS.get(feat, ""))})</span></div>
<div style="margin-top:4px">Applied to every particle, weighted by head {HN(hd["head"])}’s attention α for that particle. Fires {when}{"" if k == "lin" else f"; those are {kinds}; hard (pT share > 5 %) {100 * hd['hard']:.0f} %"}. The amount added grows with the distance from the threshold.</div></div>
<div><div class="cnt">distribution of this input over the particles (threshold in orange)</div><div class="hh" data-f="{html.escape(feat)}" data-thr="{'' if hd['thr'] is None else hd['thr']}"></div></div>
<div><div class="cnt">what this statement adds to the neuron, per jet, by true class (α-weighted)</div><table class="bars">{cls}</table></div></div></details>''')
        return ''.join(out) or '<p class="cnt">no particle input: this neuron is a constant plus the class-token terms</p>'
    def groups_html(gs):
        if not gs: return '<p class="cnt">no groups</p>'
        top_cls = lambda g: ', '.join(f'{CLASSES[c]} {v:+.2f}' for c, v in sorted(enumerate(g['by_class']), key=lambda t: -abs(t[1]))[:3])
        words = ''.join(f'<li><b>{html.escape(g["name"])}</b>: {html.escape(group_words(g, CLASSES))}</li>' for g in gs)
        rows = ''.join(f'<tr><td>{html.escape(g["name"])}</td><td class="num">{100 * g["share"]:.1f} %</td><td class="num">{g["mean_f"]:+.3g}</td><td class="num">{g["mean_w"]:.2f}</td><td class="num">{100 * g["contrib"]:+.1f} %</td><td>{", ".join(f"{k} {100 * v:.0f} %" for k, v in sorted(g["kinds"].items(), key=lambda t: -t[1]) if v > .05)}; hard {100 * g["hard"]:.0f} %</td><td>{top_cls(g)}</td></tr>' for g in gs)
        return ('<p class="cnt" style="margin:0 0 6px">Groups = the populated firing patterns of the neuron’s largest if-statements (which statements fire for a particle), the 10 largest by their share of the neuron.</p><table><tr><th>particles for which</th><th class="num">share of particles</th><th class="num">mean f (α-averaged over heads)</th><th class="num">mean weight ×n</th><th class="num">share of the neuron</th><th>what they are</th><th>jets where they add most (per jet)</th></tr>' + rows
                + '</table><ul style="margin:8px 0 4px 18px">' + words + '</ul><p class="cnt">mean weight ×n: the heads’ average weight on these particles times the jet’s multiplicity (1 = an average particle). Share of the neuron: their part of Σ_h Σ_i |α f| over all jets. Hard: pT share above 5 %.</p>')
    hpill = lambda d: ''.join(f'<span class="cnt" style="margin-right:6px">{HN(h)} {100 * s:.0f} %</span>' for h, s in sorted(enumerate(d['head_share']), key=lambda t: -t[1])[:4] if s > .05)
    dd = ''; by_imp = sorted(an['neurons'], key=lambda d: -d['strength'])
    for d in by_imp:
        fcw = ', '.join(f'{CLASSES[c]} {d["fc"][c]:+.2f}' for c in np.argsort(-np.abs(d['fc']))[:3])
        dd += f'''<details class="nd" id="neu{d["n"] + 1}"><summary><b>{NL(d["n"])}</b> — {html.escape(d["title"])} <span class="pill" style="background:{imp_col[d["importance"]]};color:{"#fff" if d["importance"] == "major" else "#1d2433"}">{d["importance"]}</span> <span class="cnt">{len(d["inputs"])} inputs · R² {d["r2"]:.2f}</span></summary>
<p style="margin:6px 0"><b>What it measures.</b> {html.escape(d["measures"])}</p><p style="margin:4px 0"><b>Role.</b> {html.escape(d["role"])} <span class="cnt">Scale: {html.escape(d["scale"])}.</span></p>
<p style="margin:4px 0" class="cnt"><b>Which heads carry it:</b> {hpill(d)} (share of the neuron’s variation coming through each head’s weights). <b>In the last layer</b> it enters the class scores with weights {html.escape(fcw)}, …</p>
<details open><summary><b>If-statements</b> <span class="cnt">(applied to each particle of the jet in turn; the neuron = b + Σ<sub>h</sub> [Σ<sub>i</sub> α<sub>hi</sub> · f<sub>h</sub>(particle i) + c<sub>h</sub> α<sub>h,cls</sub>]; when the heads share a statement it is listed once with each head’s coefficient c<sub>h</sub>; open a statement for its details)</span></summary>{terms_html(d)}</details>
<details><summary><b>Formula</b></summary><pre>{html.escape(neuron_formula(d))}</pre></details>
<details><summary><b>Python code</b> <span class="cnt">(complete, every term)</span></summary><pre>{html.escape(neuron_python(d))}</pre></details>
<details><summary><b>Particle groups</b> <span class="cnt">(this neuron as a particle tagger: the kinds of particle its if-statements single out, with what each kind does to the neuron)</span></summary>{groups_html(d["groups"])}</details>
<details><summary><b>By class</b></summary><div class="grid"><div><b>Mean value by true class</b>{bars(d["means"], "the neuron’s mean on jets of each true class", lambda v: f"{v:.2f}", center=float(np.mean(d["means"])))}</div>
<div><b>Separation (AUC vs the rest)</b>{bars(d["auc"], "0.5 = no separation", lambda v: f"{v:.2f}", center=0.5)}</div><div><b>Weight in the class scores</b>{bars(d["effect"], "change of each class score when the neuron rises by one standard deviation (its last-layer weight × its spread)", lambda v: f"{v:+.2f}")}</div></div></details>
</details>'''
    idx = ''.join(f'<a class="nbtn" href="#neu{d["n"] + 1}" style="border-left:4px solid {imp_col[d["importance"]]};text-decoration:none;color:inherit" title="{html.escape(d["title"])}" onclick="document.getElementById(\'neu{d["n"] + 1}\').open=true">{CLASSES[d["n"]] if LG else d["n"] + 1}</a>' for d in by_imp)
    # the classes: one tab per class score — which neurons go into it (links to their drop-downs), which inputs drive it
    ctabs = cpan = ''
    for i, cd in enumerate(an.get('classes', [])):
        ctabs += f'<button class="tab cb{" on" if i == 0 else ""}" onclick="setCls({i})">{CLASSES[cd["c"]]}</button>'
        nrows = ''.join(f'<tr><td><a href="#neu{x["n"] + 1}" onclick="document.getElementById(\'neu{x["n"] + 1}\').open=true">neuron {x["n"] + 1}</a></td><td>{html.escape(x["title"])}</td><td class="num">{x["w"]:+.3f}</td><td class="num">{x["spread"]:.3f}</td><td style="width:140px"><div class="bar" style="width:{120 * x["share"] / (cd["neurons"][0]["share"] or 1):.0f}px;background:{"#2f855a" if x["w"] > 0 else "#c05621"}"></div></td><td class="num">{100 * x["share"]:.0f} %</td></tr>' for x in cd['neurons'])
        irows = ''.join(f'<tr><td>{html.escape(x["feature"])}</td><td class="cnt">{html.escape(WORDS.get(x["feature"], ""))}</td><td class="num">±{x["spread"]:.3f}</td><td class="num">{x["sign"]:+.2f}</td></tr>' for x in cd['inputs'])
        cpan += f'''<div class="cpan{" on" if i == 0 else ""}"><p style="margin:0 0 6px"><b>{CLASSES[cd["c"]]} score</b> = {cd["bias"]:+.3f} + Σ<sub>n</sub> W<sub>{CLASSES[cd["c"]]},n</sub> · neuron<sub>n</sub> (ParT’s last layer). Its agreement with ParT on jets ParT calls {CLASSES[cd["c"]]}: {100 * cd["agree"]:.1f} %; separation of true {CLASSES[cd["c"]]} jets (AUC) {cd["auc"]:.3f}.</p>
<div class="grid"><div style="grid-column:span 2"><b>The neurons that go into it</b> <span class="cnt">(95 % of how much the score moves across jets; weight W × the neuron’s spread; click a neuron to open its formulas below)</span><table><tr><th>neuron</th><th>what it is</th><th class="num">weight W</th><th class="num">W × spread</th><th></th><th class="num">share</th></tr>{nrows}</table></div>
<div><b>The inputs that drive it</b> <span class="cnt">(through all its neurons and heads: spread of what each input adds to this score; correlation of that with the score)</span><table><tr><th>input</th><th>in words</th><th class="num">±</th><th class="num">corr.</th></tr>{irows}</table></div>
<div><b>Mean {CLASSES[cd["c"]]} score by true class</b>{bars(cd["means"], "the score’s mean on jets of each true class", lambda v: f"{v:.2f}", center=float(np.mean(cd["means"])))}</div></div></div>'''
    classes_html = (f'<h2>The 10 class scores</h2><div class="card"><p class="cnt" style="margin:0 0 6px">Each class score is ParT’s last layer applied to the neurons below: a weighted sum of them plus a bias. Pick a class to see which neurons go into it.</p><div>{ctabs}</div>{cpan}</div>') if ctabs else ''
    # the heads
    hrows = ''
    for i, hd in enumerate(an['heads']):
        prof = ''.join(f'<div><div class="cnt">weight vs {html.escape(k)} <small>(×average particle; dashed = 1)</small></div>{svg_profile(v)}</div>' for k, v in hd['selection'].items() if isinstance(v, dict))
        serves = ', '.join(f'<a href="#neu{n + 1}" onclick="document.getElementById(\'neu{n + 1}\').open=true">n{n + 1}</a> ({100 * s:.0f} %)' for n, s in hd['serves'])
        hrows += f'''<details class="nd"><summary><b>Block {hd["block"]}, head {hd["head"]}</b> — looks at {html.escape(hd["selects"])} <span class="cnt">· {hd["eff_particles"]:.1f} effective particles · class token keeps {100 * hd["self_weight"]:.0f} % · {hd["inputs"]} inputs, {hd["terms"]} terms in its formulas · removing its contribution: −{100 * hd["drop"]:.2f} pt</span></summary>
<p style="margin:6px 0"><b>Selects</b> {html.escape(hd["selects"])}: about {hd["eff_particles"]:.0f} effective particles per jet, the class token keeping {100 * hd["self_weight"]:.0f} % of the weight for itself (so Σ<sub>i</sub> α<sub>hi</sub> over the particles is {100 * (1 - hd["self_weight"]):.0f} % on average).
<b>Serves</b> most: neurons {serves} (share of their variation through this head).</p><div class="grid">{prof}</div>
<div class="grid" style="margin-top:6px"><div><b>Its selection by true class</b><table><tr><th>class</th><th class="num">effective particles</th><th class="num">class-token share</th></tr>{"".join(f'<tr><td>{x["cls"]}</td><td class="num">{x["eff"]:.1f}</td><td class="num">{100 * x["self"]:.1f} %</td></tr>' for x in hd["by_class"])}</table></div></div></details>'''
    FW = np.array(an['fc']['W']); fct = '<table><tr><th>class score</th><th>bias</th><th>largest positive weights (neuron: weight)</th><th>largest negative weights</th></tr>'
    for c in range(10):
        o = np.argsort(-FW[c]); fct += f'<tr><td>{CLASSES[c]}</td><td class="num">{an["fc"]["b"][c]:+.2f}</td><td>{", ".join(f"n{n + 1}: {FW[c, n]:+.2f}" for n in o[:6])}</td><td>{", ".join(f"n{n + 1}: {FW[c, n]:+.2f}" for n in o[::-1][:6])}</td></tr>'
    fct += '</table>'
    jb = ''.join(f'<span class="btn jb{" on" if i == 0 else ""}" onclick="setJet({i})" style="border-color:{"#2f855a" if e["model"] == e["part"] else "#c05621"}">{CLASSES[e["part"]]} <small>{e["n"]}p</small></span>' for i, e in enumerate(an['examples']))
    js = """const EX=__EX__, CL=__CL__, HS=__HS__, FC=__FC__, TT=__TT__, HNM=__HNM__; let JJ=0, NEU=0;
function histSVG(hs,thr){const w=260,h=70,e=hs.edges,c=hs.counts,mx=Math.max(...c)||1,n=c.length,bw=(w-20)/n,X=v=>10+(w-20)*(v-e[0])/(e[e.length-1]-e[0]+1e-12);
 let g='';c.forEach((v,i)=>{g+='<rect x="'+(10+i*bw).toFixed(1)+'" y="'+(h-14-(h-22)*v/mx).toFixed(1)+'" width="'+(bw-1).toFixed(1)+'" height="'+((h-22)*v/mx).toFixed(1)+'" fill="#c9d4e3"/>';});
 if(thr!==''){const t=parseFloat(thr);g+='<line x1="'+X(t).toFixed(1)+'" x2="'+X(t).toFixed(1)+'" y1="4" y2="'+(h-12)+'" stroke="#c05621" stroke-width="2"/><text x="'+(X(t)+3).toFixed(1)+'" y="12" font-size="10" fill="#c05621">'+t.toPrecision(3)+'</text>';}
 return '<svg viewBox="0 0 '+w+' '+h+'" style="width:100%;max-width:'+w+'px">'+g+'<text x="10" y="'+(h-2)+'" font-size="9" fill="#667">'+e[0].toPrecision(3)+'</text><text x="'+(w-10)+'" y="'+(h-2)+'" font-size="9" text-anchor="end" fill="#667">'+e[e.length-1].toPrecision(3)+'</text></svg>';}
document.addEventListener('toggle',ev=>{const d=ev.target; if(!d.classList||!d.classList.contains('xterm')||!d.open) return; d.querySelectorAll('.hh').forEach(x=>{if(!x.dataset.done){x.innerHTML=histSVG(HS[x.dataset.f],x.dataset.thr);x.dataset.done=1;}});},true);
function setJet(i){JJ=i;document.querySelectorAll('.jb').forEach((e,k)=>e.classList.toggle('on',k==i));drawJet();}
function setNeu(n){NEU=n;document.getElementById('neusel').value=n;drawJet();}
function jetSVG(e,c){const W=360,Hh=300,R=0.8,X=v=>W/2+v/R*(W/2-20),Y=v=>Hh/2-v/R*(Hh/2-20);const mx=Math.max(...c.map(Math.abs))||1;
 let g='<svg viewBox="0 0 '+W+' '+Hh+'" style="width:100%;max-width:'+W+'px;background:#fbfcfe;border:1px solid #e4e7ec;border-radius:8px">';
 g+='<line x1="'+X(-R)+'" x2="'+X(R)+'" y1="'+Y(0)+'" y2="'+Y(0)+'" stroke="#eee"/><line y1="'+Y(-R)+'" y2="'+Y(R)+'" x1="'+X(0)+'" x2="'+X(0)+'" stroke="#eee"/>';
 [0.2,0.4,0.8].forEach(r=>{g+='<circle cx="'+X(0)+'" cy="'+Y(0)+'" r="'+(r/R*(W/2-20))+'" fill="none" stroke="#e4e7ec" stroke-dasharray="2 3"/><text x="'+(X(r)+2)+'" y="'+(Y(0)-2)+'" font-size="9" fill="#99a">ΔR '+r+'</text>';});
 e.particles.forEach((p,k)=>{const a=Math.abs(c[k])/mx, r=3+10*Math.sqrt(p.z), col=c[k]>=0?'47,133,90':'192,86,33'; g+='<circle cx="'+X(p.eta)+'" cy="'+Y(p.phi)+'" r="'+r.toFixed(1)+'" fill="rgba('+col+','+(0.15+0.85*a).toFixed(2)+')" stroke="#889" stroke-width="0.5" style="cursor:pointer" onclick="showP('+k+')"><title>particle '+(k+1)+': adds '+c[k].toFixed(3)+' to neuron '+(NEU+1)+', pT share '+p.z.toFixed(3)+', '+p.type+'</title></circle>';});
 g+='<text x="8" y="'+(Hh-8)+'" font-size="10" fill="#667">Δη →   (Δφ ↑) · size: pT share · green/orange: adds to / subtracts from neuron '+(NEU+1)+' (darker = more)</text></svg>'; return g;}
function drawJet(){const e=EX[JJ], c=e.C.map(r=>r[NEU]); let tot=0; c.forEach(v=>tot+=v); const abar=e.particles.map((p,k)=>{let s=0;for(let h=0;h<16;h++)s+=e.A[h][k];return s/16;});
 let s='<div class="cnt">ParT: <b>'+CL[e.part]+'</b> ('+(100*e.p_part[e.part]).toFixed(0)+'%) · this model: <b>'+CL[e.model]+'</b> ('+(100*e.p_model[e.model]).toFixed(0)+'%) · truth '+CL[e.truth]+' · '+e.n+' particles. Showing <b>neuron '+(NEU+1)+'</b>: '+TT[NEU]+'. Click a particle for its 16 weights and its inputs.</div>';
 s+='<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;margin:8px 0"><div>'+jetSVG(e,c)+'</div><div class="cnt" style="max-width:380px"><b>How this neuron is computed for this jet.</b> Each of the 16 heads weights the particles in its own way (α, ParT’s); the neuron adds, for every head, Σ<sub>i</sub> α<sub>hi</sub> f<sub>h</sub>(particle i). The table gives each particle’s total over the heads, g<sub>i</sub> = Σ<sub>h</sub> α<sub>hi</sub> f<sub>h</sub>(x<sub>i</sub>), and the mean of its 16 weights. <b>How α is computed:</b> by ParT. Its particle blocks give each particle an embedding e<sub>i</sub> (the particle in its jet context); then α<sub>hi</sub> = exp(s<sub>hi</sub>) / Σ<sub>j</sub> exp(s<sub>hj</sub>) with s<sub>hi</sub> = q<sub>h</sub> · W<sub>k,h</sub> LN(e<sub>i</sub>) / 4, q<sub>h</sub> the class token’s query (block 1: the same for every jet; block 2: from block 1’s output), j over the class token and the particles. This is the one part of this model that is not a formula.</div></div>';
 s+='<table><tr><th>#</th><th class="num">pT share</th><th class="num">ΔR</th><th>type</th><th class="num">charge</th><th class="num">|d0|/σ</th><th class="num">mean α (16 heads)</th><th class="num">g<sub>i</sub> → neuron '+(NEU+1)+'</th><th></th></tr>';
 const mx=Math.max(...c.map(Math.abs))||1; e.particles.forEach((p,k)=>{s+='<tr class="p" onclick="showP('+k+')"><td>'+(k+1)+'</td><td class="num">'+p.z.toFixed(3)+'</td><td class="num">'+p.dr.toFixed(2)+'</td><td>'+p.type+'</td><td class="num">'+(p.q>0?'+':'')+p.q+'</td><td class="num">'+p.d0s.toFixed(1)+'</td><td class="num">'+abar[k].toFixed(3)+'</td><td class="num">'+c[k].toFixed(3)+'</td><td><span class="wbar" style="width:'+(120*Math.abs(c[k])/mx)+'px;background:'+(c[k]>=0?'#2f855a':'#c05621')+'"></span></td></tr>';});
 s+='</table><pre>neuron '+(NEU+1)+' = Σ_i g_i  +  Σ_h c_h · α_h,cls  +  b  =  '+tot.toFixed(3)+'  +  '+(e.V[NEU]-tot).toFixed(3)+'  =  '+e.V[NEU].toFixed(3)+'      (ParT’s own neuron '+(NEU+1)+' on this jet: '+e.H[NEU].toFixed(3)+')</pre>';
 let sc='<div class="cnt" style="margin-top:6px"><b>Class scores from the 128 neurons</b> (ParT’s last layer): score<sub>c</sub> = b<sub>c</sub> + Σ<sub>n</sub> W<sub>cn</sub> · neuron<sub>n</sub>. This jet: '+CL.map((nm,ci)=>{let v=FC.b[ci];for(let n=0;n<e.V.length;n++)v+=FC.W[ci][n]*e.V[n];return nm+' '+v.toFixed(2);}).join(' · ')+'.</div>';
 document.getElementById('jet').innerHTML=s+sc; document.getElementById('contrib').innerHTML='<span class="cnt">click a particle</span>';}
function showP(k){const e=EX[JJ]; document.querySelectorAll('#jet tr.p').forEach((r,i)=>r.classList.toggle('on',i==k));
 let t='<div class="cnt">particle '+(k+1)+' — its weight α in each head (ParT’s), and its inputs x<sub>i</sub> (what the formulas are evaluated on):</div><table><tr>'+HNM.map(h=>'<th class="num">'+h+'</th>').join('')+'</tr><tr>'+e.A.map(a=>'<td class="num">'+a[k].toFixed(3)+'</td>').join('')+'</tr></table>';
 t+='<table style="margin-top:6px">'+NAMES.map((nm,f)=>'<tr><td>'+nm+'</td><td class="num">'+e.X[k][f].toPrecision(4)+'</td></tr>').join('')+'</table>'; document.getElementById('contrib').innerHTML=t;}
function setCls(i){document.querySelectorAll('.cpan').forEach((e,k)=>e.classList.toggle('on',k==i));document.querySelectorAll('.cb').forEach((e,k)=>e.classList.toggle('on',k==i));}
const NAMES=__NAMES__; setJet(0);""".replace('__EX__', json.dumps(an['examples'])).replace('__CL__', json.dumps(CLASSES)).replace('__HS__', json.dumps(an['hists'])).replace('__FC__', json.dumps(an['fc'])) \
        .replace('__TT__', json.dumps([d['title'] for d in an['neurons']])).replace('__HNM__', json.dumps([HN(h) for h in range(16)])).replace('__NAMES__', json.dumps((PFEAT + NBR)[:NV]))
    neusel = '<select id="neusel" onchange="setNeu(parseInt(this.value))">' + ''.join(f'<option value="{d["n"]}">neuron {d["n"] + 1} — {html.escape(d["title"])}</option>' for d in an['neurons']) + '</select>'
    NOT = [('jet, particles', 'a jet has n particles i = 1…n (up to 128); x_i = the 38 per-particle inputs of particle i (defined in the table at the bottom).'),
           ('heads h', 'ParT’s 2 class-attention blocks × 8 heads = 16 heads, h = b1h1 … b2h8. Each head weights the particles of a jet in its own way.'),
           ('attention weight α_hi', 'head h’s weight of particle i: α_hi = exp(s_hi) / Σ_j exp(s_hj), a softmax over [the class token (i = 0), the particles]; the n + 1 weights of a head sum to 1. Computed by ParT from its particle embeddings (not a formula here). α_h,cls = α_h0, the class token’s own share.'),
           ('terms t(x)', 'for every input x_f: x_f itself, max(0, x_f − θ) or max(0, θ − x_f) at one of 5 thresholds θ (the 15–85 % quantiles of x_f over particles). At most one term per input, head and neuron.'),
           ('neuron_n', 'n = 1…128: neuron_n = b_n + Σ_h [ Σ_i α_hi · f_hn(x_i) + c_hn · α_h,cls ], with f_hn(x) = Σ_terms w · t(x). These are ParT’s 128 class-token neurons (after its final LayerNorm), each written as per-particle formulas.'),
           ('class scores', 'ParT’s last layer: score_c = b_c + Σ_n W_cn · neuron_n (a fixed 10 × 128 linear map); the class = the largest score; probabilities = softmax of the scores.')]
    notation_rows = ''.join(f'<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(v)}</td></tr>' for k, v in NOT)
    defs_rows = ''.join(f'<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(WORDS.get(k, ""))}</td><td class="cnt">{html.escape(v)}</td></tr>' for k, v in DEFS.items() if k in (PFEAT + NBR)[:NV])
    (outdir / f'{tag}_formulas.py').write_text(python_export(tag, an)); shutil.copy(OUT / f'{tag}_model.npz', outdir / f'{tag}_model.npz')
    (outdir / 'analysis.json').write_text(json.dumps({k: v for k, v in an.items() if k not in ('examples', 'hists')}))
    compose = """for every jet:
    X[i, :]            = the 38 per-particle inputs of particle i (definitions at the bottom of the page)
    alpha[h, :]        = ParT's attention weights of head h over [class token, particles], h = 1..16      # kept from ParT
    for n in 1..128:
        neuron[n]      = b_n + sum_h ( sum_i alpha[h, i] * f_hn(X[i, :]) + c_hn * alpha[h, cls] )           # FORMULAS: one f per head and neuron
                         f_hn(x) = sum over its inputs of w * t(x),  t(x) = x_f | max(0, x_f - θ) | max(0, θ - x_f)
    class scores       = FC_W @ neuron + FC_B                                                                 # ParT's last layer (fixed 10 x 128)"""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}.cpan{{display:none}}.cpan.on{{display:block}}</style></head><body><main>
<p class="cnt"><a href="../../index.html">← all setups</a> · <a href="../../research/index.html">research log</a></p><h1>{html.escape(title)}</h1>
<p class="cnt">ParT_full · JetClass. The 128 numbers ParT’s class token ends with — the neurons that its last layer turns into the class scores — each written directly as per-particle formulas: for every head, a formula of each particle’s physics, summed over the particles with that head’s attention weights. Nothing of ParT in between: no per-head values, no out-projection, no MLPs, no LayerNorms.</p>
<div>{kpi}</div>
<div class="card"><div class="keep"><b>Kept of ParT:</b> {kept}</div><div class="form"><b>Formulas:</b> {formula}</div></div>
<h2>Notation</h2><div class="card"><table>{notation_rows}</table></div>
<h2>What the output is composed of</h2><div class="card"><pre>{html.escape(compose)}</pre><p><b>How the per-particle formulas become a neuron.</b> Every neuron has one formula per head, f<sub>hn</sub>. Head h weights the particles of the jet with its α (ParT’s), and the neuron adds Σ<sub>i</sub> α<sub>hi</sub> f<sub>hn</sub>(x<sub>i</sub>) for each of the 16 heads, plus c<sub>hn</sub> times the class token’s own share of that head, plus a bias. So one neuron is 16 weighted averages of particle properties — the same 16 selections for all 128 neurons, 128 different things read off them.</p>
<p><b>How the 128 neurons become the class scores.</b> ParT’s last layer, kept as it is: score<sub>c</sub> = b<sub>c</sub> + Σ<sub>n</sub> W<sub>cn</sub> · neuron<sub>n</sub>; the class is the largest score. The weights W (10 × 128):</p>{fct}
<p class="cnt">Exact formulation with the fitted coefficients: <a href="{tag}_formulas.py">{tag}_formulas.py</a> (+ <a href="{tag}_model.npz">{tag}_model.npz</a>). Every neuron is written out term by term below.</p></div>
<h2>The 16 heads: who weights the particles</h2><div class="card"><p class="cnt" style="margin:0 0 6px">The attention weights are ParT’s (its particle blocks → embeddings → class-token query · keys → softmax); per head and per jet. What each head selects on average, and which neurons it carries:</p>{hrows}</div>
<h2>A jet, step by step</h2><div class="card"><div class="cnt">jet (6 per ParT class; green border = this model agrees with ParT on it):</div><div>{jb}</div><div style="margin-top:6px"><span class="cnt">neuron to follow:</span> {neusel}</div></div><div class="card" id="jet"></div><div class="card" id="contrib"></div>
{classes_html}<h2>{"The 10 class scores, each written directly" if LG else "The 128 neurons"}</h2><div class="card"><p style="margin:0 0 6px"><b>{len(act)} of the {len(an["neurons"])} {"class scores" if LG else "neurons"} have particle inputs after pruning</b> (neurons {', '.join(str(d['n'] + 1) for d in act)}); the others are constants plus class-token terms that hardly vary, and ParT’s last layer gives them little weight (norm of its weights on the active neurons {float(np.linalg.norm(np.array(an['fc']['W'])[:, [d['n'] for d in act]])):.1f} vs {float(np.linalg.norm(np.delete(np.array(an['fc']['W']), [d['n'] for d in act], 1))):.1f} on the rest). <b>How to read a neuron:</b> neuron = b + Σ over the 16 heads h of [ Σ over <b>every particle i of the jet</b> of α<sub>hi</sub> · f<sub>h</sub>(particle i) + c<sub>h</sub> α<sub>h,cls</sub> ]. f<sub>h</sub> is the formula written under “head b·h·” in the statements, evaluated on that particle’s inputs; α<sub>hi</sub> is head h’s weight for that particle in this jet. Importance: how much the neuron moves the class scores (its last-layer weights × its spread), ranked over the 128 — major = top 20 %, minor = bottom half. R²: how well the formula neuron tracks ParT’s own neuron across jets.</p>
<div style="margin:4px 0 8px"><span class="cnt">neurons, most important first (jump to):</span> {idx}</div>{dd}</div>
{metrics_html}
<h2>Definitions of the per-particle inputs</h2><div class="card"><table><tr><th>input</th><th>in words</th><th>definition (math)</th></tr>{defs_rows}</table><p class="cnt">Every input describes <b>one particle i</b> of the jet — the particle a head is summing over. “Nearest” always means nearest <b>to particle i</b>, among the <b>other particles of the same jet</b>, by the angle ΔR = √(Δη² + Δφ²).</p><p class="cnt">Code: jetdistill/part/clsfit.py (particle_features), jetdistill/research/nbr.py (nbr_features).</p></div>
<p class="cnt">Analysis: <a href="analysis.json">analysis.json</a>. Research log: <a href="../../research/index.html">research page</a>.</p>
</main><script>{js}</script></body></html>"""
    (outdir / 'index.html').write_text(doc); log(f'page: {outdir / "index.html"} {len(doc) // 1024} kB')


if __name__ == '__main__':
    page(sys.argv[1], sys.argv[2])
