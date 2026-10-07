"""The page of a per-head formula model (S5 / S8_uniform2: ParT's selection kept; S10*: all formulas). Built around
the rule each head applies to a jet: (1) a jet explorer — pick a jet and a head, see every particle's score and
attention weight (the class token's own share too) and, per particle, which physics set its score; (2) the rule
itself per head — net response to each input, measured weight profiles, importance; (3) the 16 value neurons per head
as formulas; (4) what is kept of ParT; (5) the paper's metrics vs ParT and the jet-level formula, per class.

  python -m jetdistill.research.heads_page S5|S8_uniform2|S10|S10b OUT_DIR"""
import html, json, shutil, sys, pathlib
import numpy as np
from ..config import RESULTS, CLASSES
from ..pipeline import jets
from ..part.clsfit import PFEAT, particle_features
from ..part.network import ParTNetwork
from .nbr import NBR, PK, nbr_features, pk_features
from .heads import extract, downstream, phi_pooled, rows_of_split, OUT
from .heads_eval import uniformize
from .heads_defs import DEFS, WORDS, PK_DEF, COMPOSE, NOTATION, PHRASE, python_export, neuron_snippet, if_lines, neuron_python, group_words

CSS = """:root{--ink:#1d2433;--mut:#667085;--line:#e4e7ec;--bg:#f7f8fa;--acc:#1f4e79;--good:#2f855a;--bad:#c05621}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:18px 16px 60px}h1{font-size:22px;margin:0 0 6px}h2{font-size:17px;margin:22px 0 8px}h3{font-size:15px;margin:14px 0 6px}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0;overflow-x:auto}
.cnt{color:var(--mut);font-size:12.5px}table{border-collapse:collapse}td,th{padding:3px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-size:13px}
th{font-weight:600;background:#f2f4f7}.num{text-align:right;font-variant-numeric:tabular-nums}a{color:var(--acc)}
.kpi{display:inline-block;background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px 12px;margin:4px 8px 4px 0;min-width:150px}.kpi b{font-size:20px;display:block}
.keep{border-left:4px solid var(--bad);background:#fff7f2;padding:8px 12px;border-radius:6px}.form{border-left:4px solid var(--good);background:#f2fbf5;padding:8px 12px;border-radius:6px;margin-top:6px}
.btn{display:inline-block;border:1px solid #c9ced6;border-radius:8px;padding:3px 9px;margin:2px;cursor:pointer;background:#fff;font-size:12.5px}.btn.on{outline:2px solid var(--acc)}.btn small{color:var(--mut)}
.pill{display:inline-block;border-radius:10px;padding:0 7px;font-size:11.5px;color:#fff;background:#667}.wbar{display:inline-block;height:10px;background:var(--acc);border-radius:2px;vertical-align:middle}
tr.p{cursor:pointer}tr.p:hover{background:#f2f6fb}tr.p.on{background:#e8f0fa}tr.p.core td:first-child{border-left:3px solid #c05621;font-weight:600}tr.cls td{background:#fffaf0;font-style:italic}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}.bars td{padding:2px 8px}.bar{height:10px;background:var(--acc);border-radius:3px}
.head{display:none}.head.on{display:block}details.nd{border-top:1px solid var(--line);padding:6px 0}details.xterm{border-top:1px dashed var(--line);padding:3px 0;margin-left:10px}details.xterm>summary{cursor:pointer}details.xterm code{font:12px ui-monospace,Menlo,monospace;background:#f3f5f8;padding:1px 4px;border-radius:4px}details.nd>summary{cursor:pointer;font-size:14px}details details{margin:4px 0 4px 14px}details details>summary{cursor:pointer}#bar{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:6px 0 8px;margin-bottom:8px}.tab{border:none;background:none;padding:6px 10px;font:inherit;font-weight:600;color:var(--mut);cursor:pointer;border-bottom:3px solid transparent}.tab.on{color:var(--acc);border-bottom-color:var(--acc)}.tab small{font-weight:400}.nbtn{display:inline-block;border:1px solid #c9ced6;border-radius:6px;padding:1px 6px;margin:2px;cursor:pointer;background:#fff;font-size:12px}.nbtn.on{background:var(--acc);color:#fff;border-color:var(--acc)}.neu{display:none}.neu.on{display:block}
pre{font:12px/1.55 ui-monospace,Menlo,monospace;background:#f6f8fa;border:1px solid var(--line);border-radius:8px;padding:8px 10px;overflow-x:auto;white-space:pre-wrap;margin:4px 0}#contrib{min-height:60px}"""

INFO = {
    'S5': ('Per-head formulas, ParT’s selection', 'Both class blocks’ attention weights (which particles each head looks at, per jet) are ParT’s, computed from its particle blocks.',
           'Every head’s value — what it sums over the particles it attends to — is a formula of each particle’s own physics.'),
    'S8_uniform2': ('Per-head formulas, ParT’s block-1 selection', 'Class block 1’s attention weights are ParT’s (from its particle blocks); block 2 is a plain average over the particles.',
                    'Every head’s value is a formula of each particle’s own physics; block 2 needs no selection.'),
    'S10': ('All formulas', 'Nothing above the class attention: no particle blocks, no embeddings.',
            'Block 1’s selection (per head one score formula of a particle’s physics; per jet, softmax over the class token’s fixed score and the particles’ scores) and every head’s value are formulas; block 2 is a plain average.')}
INFO['S10b'] = INFO['S10']
INFO['S5rp'] = ('Per-head formulas, pruned — ParT’s selection kept', INFO['S5'][1], INFO['S5'][2] + ' Pruned: each head keeps only the inputs it needs (17–29 of 38), re-tuned.')
INFO['S5r'] = INFO['S5']
INFO['S5rpc1'] = ('Per-head formulas, one term per input — ParT’s selection kept', INFO['S5'][1], INFO['S5'][2] + ' Every input enters each value neuron through at most one term (x, max(0, x − θ) or max(0, θ − x)), re-tuned.')
INFO['S5rpc'] = ('Per-head formulas, pruned, low cancellation — ParT’s selection kept', INFO['S5'][1], INFO['S5'][2] + ' Pruned and re-tuned so that inputs do not offset each other.')
DOWN = 'Kept in every model: ParT’s fixed arithmetic after the heads — out-projection, per-head scale, LayerNorms, the 128→512→128 MLP with residuals in both class blocks, the final LayerNorm and the last layer. Not fitted, no physics content: a smooth map from the 256 head outputs to the 10 class scores.'
TYPES = ['ch. hadron', 'n. hadron', 'photon', 'electron', 'muon']


def load_model(tag):
    Pz = dict(np.load(OUT / f'{tag}_model.npz')); names = PFEAT + NBR + PK
    if tag.startswith('S10'):
        nv = Pz['Wv'].shape[1] // 6
        return dict(kind='joint', names=names, nv=nv, **{k: Pz[k] for k in ('Ws', 'Wv', 'cself', 'bias', 'kns', 'knv', 'mus', 'sds', 'muv', 'sdv')})
    if str(Pz.get('basis', '')) == 'extended':
        from .one_term import to_standard
        nv = int(Pz['nv']); return dict(kind='heads', W=to_standard(Pz['W'], Pz['knv'], nv), Wext=Pz['W'], knv=Pz['knv'], kn=Pz['kn'], uniform=str(Pz['uniform']), names=names, nv=nv)
    return dict(kind='heads', W=Pz['W'], kn=Pz['kn'], uniform=str(Pz['uniform']), names=names, nv=38)


def analyze(tag, n_dev=20000, n_ex=6, device='mps', log=print):
    import torch
    m = load_model(tag); model = ParTNetwork('full').model; model.eval(); J = jets('full', 'dev'); rows = rows_of_split('dev', n_dev)
    _, A, M, L, _ = extract(model, 'dev', n_dev, device); ref = L.argmax(1); T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)
    nv, names = m['nv'], m['names']; vnames = names[:nv]; nf = len(names)
    F0, ok = particle_features(J, rows, ctx=[]); Fn = nbr_features(J['x'][rows], J['ext'][rows], J['jet'][rows]); Fp = pk_features(J['x'][rows], J['ext'][rows], J['jet'][rows], model, device)
    F = np.concatenate([F0, Fn, Fp], -1); del Fp
    if m['kind'] == 'heads':
        A = uniformize(A, M, [int(c) - 1 for c in m['uniform']]); P, _ = phi_pooled(J, rows, A, m['kn']); W = m['W']
        O = np.einsum('nhk,hko->nho', P, W); Pv = P[:, :, :6 * nv]; Wv = W[:, :6 * nv]; kn = m['kn'][:, :nv]; mu = np.zeros(nv); sd = np.ones(nv); S = None
    else:
        from .joint import Model
        mdl = Model(dict(kn=m['kns'], mu=m['mus'], sd=m['sds']), m['knv'], m['muv'], m['sdv'], device, m.get('ms'), m.get('mv')); Ws, Wv_t, cs, b = T(m['Ws']), T(m['Wv']), T(m['cself']), T(m['bias'])
        Os, Pvs, Ss = [], [], []
        with torch.no_grad():
            for a in range(0, n_dev, 2000):
                Ft, mt = torch.from_numpy(F[a:a + 2000]).to(device).half().float(), torch.from_numpy(ok[a:a + 2000]).to(device); n = Ft.shape[0]
                Os.append(mdl.heads(Ft, mt, Ws, Wv_t, cs, b).cpu().numpy()); s = (mdl.phis(Ft) @ Ws).masked_fill(~mt[..., None], -1e9); Ss.append(s.cpu().numpy())
                a1 = torch.softmax(torch.cat([torch.zeros(n, 1, 8, device=device), s], 1), 1); cnt = mt.sum(1, keepdim=True).float() + 1; a2 = (mt.float() / cnt)[..., None].expand(-1, -1, 8); pv = mdl.phiv(Ft)
                Pvs.append(torch.cat([torch.einsum('nph,npk->nhk', a1[:, 1:], pv), torch.einsum('nph,npk->nhk', a2, pv)], 1).cpu().numpy())
                A[0, a:a + n] = np.concatenate([a1.cpu().numpy()[:, 0][:, :, None], np.zeros((n, 8, 128), np.float32)], -1); A[0, a:a + n, :, 1:1 + Ft.shape[1]] = a1.cpu().numpy()[:, 1:].transpose(0, 2, 1)
                A[1, a:a + n] = 0; A[1, a:a + n, :, 0] = (1 / cnt).expand(-1, 8).cpu().numpy(); A[1, a:a + n, :, 1:1 + Ft.shape[1]] = a2.cpu().numpy().transpose(0, 2, 1)
        O = np.concatenate(Os); Pv = np.concatenate(Pvs); S = np.concatenate(Ss); Wv = m['Wv']; kn = m['knv']; mu, sd = m['muv'], m['sdv']
    def logits(o):
        with torch.no_grad(): return np.concatenate([downstream(model, *T(o[a:a + 5000]).split(8, 1)).cpu().numpy() for a in range(0, n_dev, 5000)])
    Lm = logits(O); pred = Lm.argmax(1); a0 = float((pred == ref).mean()); heads = []
    d0s = np.abs(np.asarray(J['ext'][rows][..., 3], np.float32)) / np.maximum(np.asarray(J['ext'][rows][..., 4], np.float32), 1e-6)
    npart = ok.sum(1); ytrue = J['y'][rows]
    if m['kind'] == 'heads':                                                 # every real particle of the dev jets once: inputs, terms, weights, kind, jet class
        XR = np.concatenate([F0, Fn], -1)[..., :nv][ok]; TR = np.concatenate([XR] + [np.maximum(0, XR - kn[k][:nv]) for k in range(5)], -1)
        AR = [A[b_][:, :, 1:].transpose(0, 2, 1)[ok] for b_ in range(2)]; YR = np.repeat(ytrue, ok.sum(1)); nmean = ok.sum(1).mean()
        tp = F0[..., 8:13][ok]; KR = {'charged hadron': tp[:, 0], 'neutral hadron': tp[:, 1], 'photon': tp[:, 2], 'lepton': tp[:, 3] + tp[:, 4]}; ZR = (np.exp(F0[..., 2][ok]) > .05).astype(float)
        HIST = []
        for f in range(nv):                                                 # histogram of each input over the real particles (1–99 % range)
            lo_, hi_ = np.quantile(XR[:, f], [.01, .99]); hi_ = hi_ if hi_ > lo_ else lo_ + 1; cnts, edges = np.histogram(np.clip(XR[:, f], lo_, hi_), bins=30, range=(lo_, hi_))
            HIST.append(dict(edges=[round(float(e), 5) for e in edges], counts=[int(c) for c in cnts]))
    for h in range(16):
        o = O.copy(); o[:, h] = O[:, h].mean(0); drop = a0 - float((logits(o).argmax(1) == ref).mean())
        imp_t = np.abs(Wv[h]) * Pv[:, h].std(0)[:, None]; imp_f = np.zeros(nv)
        for j in range(Wv.shape[1]): imp_f[j % nv] += imp_t[j].sum()
        top = np.argsort(-imp_f)[:6]; curves = {}
        for f in top:
            x = np.linspace(kn[0, f] - (kn[-1, f] - kn[0, f]) * .3, kn[-1, f] + (kn[-1, f] - kn[0, f]) * .3, 40); xs = (x - mu[f]) / sd[f]
            basis = np.stack([xs] + [np.maximum(0, x - kn[k, f]) / sd[f] for k in range(len(kn))], 1); cols = [f] + [nv + k * nv + f for k in range(len(kn))]
            y = basis @ Wv[h][cols]; y -= y.mean(0); curves[vnames[f]] = dict(x=x.round(4).tolist(), y=y.round(4).T.tolist())
        neurons = []
        for o_ in range(16):                                                    # per input: its terms (symbolic), its net typical contribution
            w = Wv[h][:, o_]; ins = []
            for f in range(nv):
                cols = [f] + [nv + k * nv + f for k in range(len(kn))]
                if 'Wext' in m:                                                 # one-term model: the single term as stored
                    we = m['Wext'][h][:, o_]; terms = []
                    if we[f]: terms.append((float(we[f]), 'lin', None))
                    for k in range(5):
                        if we[nv + k * nv + f]: terms.append((float(we[nv + k * nv + f]), 'gt', float(m['knv'][k, f])))
                        if we[6 * nv + k * nv + f]: terms.append((float(we[6 * nv + k * nv + f]), 'lt', float(m['knv'][k, f])))
                else:
                    terms = ([(float(w[f]), 'lin', None)] if w[f] else []) + [(float(w[nv + k * nv + f]), 'gt', float(kn[k, f])) for k in range(len(kn)) if w[nv + k * nv + f]]
                if not terms: continue
                stats = []
                if m['kind'] == 'heads':
                    xf = XR[:, f]; awt = AR[h // 8][:, h % 8]
                    for cf, k_, t_ in terms:
                        fire = np.ones(len(xf), bool) if k_ == 'lin' else (xf > t_) if k_ == 'gt' else (xf < t_)
                        val = cf * (xf if k_ == 'lin' else np.maximum(0, xf - t_) if k_ == 'gt' else np.maximum(0, t_ - xf)); con = awt * val
                        byc = [float(con[YR == c].sum() / max((ytrue == c).sum(), 1)) for c in range(10)]
                        stats.append(dict(fires=float(fire.mean()), kinds={k2: float(v2[fire].mean()) if fire.any() else 0.0 for k2, v2 in KR.items()}, hard=float(ZR[fire].mean()) if fire.any() else 0.0, by_class=byc,
                                          hist=HIST[f], thr=None if k_ == 'lin' else float(t_)))
                ins.append(dict(feature=vnames[f], importance=float((Pv[:, h, cols] @ w[cols]).std()), terms=terms, stats=stats))
            ins.sort(key=lambda d: -d['importance']); neurons.append(ins)
        b = h // 8; al = A[b, :, h % 8, 1:]; rel = al * npart[:, None]; sel = {}
        for name, v, bins in (('ln pT/pT_jet', F0[..., 2], np.linspace(-7, -0.5, 9)), ('ΔR', F0[..., 4], np.linspace(0, 0.8, 9)), ('|d0|/σ', d0s, np.array([0, .5, 1, 2, 3, 5, 10, 1e9]))):
            idx = np.clip(np.searchsorted(bins, v) - 1, 0, len(bins) - 2); sel[name] = dict(edges=[float(e) if e < 1e8 else None for e in bins], weight=[float(rel[ok & (idx == i)].mean()) if (ok & (idx == i)).any() else None for i in range(len(bins) - 1)])
        for name, msk in (('charged', F0[..., 7] != 0), ('photon', F0[..., 10] > 0), ('lepton', (F0[..., 11] + F0[..., 12]) > 0)): sel[name] = float(rel[ok & msk].mean()) if (ok & msk).any() else None
        rule = None
        if S is not None and b == 0:                                            # the score formula: net response to each input, inputs by weight
            W_s = m['Ws'][:, h]; imp_s = np.zeros(nf); sdp = F[ok].std(0)
            for j in range(1, len(W_s)): imp_s[(j - 1) % nf] += abs(W_s[j]) * (sdp[(j - 1) % nf] / m['sds'][(j - 1) % nf])
            ts = np.argsort(-imp_s)[:8]; rc = {}
            for f in ts:
                x = np.linspace(np.quantile(F[ok][:, f], .02), np.quantile(F[ok][:, f], .98), 40); basis = np.stack([(x - m['mus'][f]) / m['sds'][f]] + [np.maximum(0, x - m['kns'][k, f]) / m['sds'][f] for k in range(len(m['kns']))], 1)
                cols = [1 + f] + [1 + nf + k * nf + f for k in range(len(m['kns']))]; y = basis @ W_s[cols]; rc[names[f]] = dict(x=x.round(4).tolist(), y=(y - y.mean()).round(4).tolist())
            rule = dict(inputs=[dict(feature=names[f], importance=float(imp_s[f])) for f in ts], curves=rc, intercept=float(W_s[0]))
        Wcb = m['Wext'][h] if 'Wext' in m else (m['W'][h] if m['kind'] == 'heads' else None)
        cb = [(float(Wcb[-2, o_]), float(Wcb[-1, o_])) for o_ in range(16)] if Wcb is not None else [(0.0, 0.0)] * 16
        heads.append(dict(cb=cb, block=b + 1, head=h % 8 + 1, drop=drop, self_weight=float(A[b, :, h % 8, 0].mean()), eff_particles=float(np.exp(-(al * np.log(al + 1e-12)).sum(1)).mean()),
                          features=[dict(feature=vnames[f], importance=float(imp_f[f])) for f in top], curves=curves, neurons=neurons, selection=sel, rule=rule))
        log(f'  block {b + 1} head {h % 8 + 1}: ablation −{100 * drop:.2f} pt; values use ' + ', '.join(vnames[f] for f in top[:4]))
    # per value neuron, facts in the style of the JEDI-linear explorer
    from sklearn.metrics import roc_auc_score
    ytrue = J['y'][rows]; Lm_ = Lm
    def head_sel_phrase(hd):
        sel = hd['selection']; out = []
        w = [v for v in sel['ln pT/pT_jet']['weight'] if v is not None]
        if len(w) > 2 and w[-1] > 1.6 * max(w[0], 1e-9): out.append('the hard particles')
        w = [v for v in sel['ΔR']['weight'] if v is not None]
        if len(w) > 2 and w[0] > 1.4 * max(w[-1], 1e-9): out.append('particles near the axis')
        elif len(w) > 2 and w[-1] > 1.4 * max(w[0], 1e-9): out.append('wide-angle particles')
        for k, nm, th in (('lepton', 'leptons', 1.5), ('charged', 'charged particles', 1.15), ('photon', 'photons', 1.2)):
            if sel.get(k) and sel[k] > th: out.append(nm)
        w = [v for v in sel['|d0|/σ']['weight'] if v is not None]
        if len(w) > 2 and w[-1] > 1.5 * max(w[0], 1e-9): out.append('displaced tracks')
        return ', '.join(out[:3]) if out else 'all particles about equally'
    def direction(d):
        """+1 if the input's piece rises with the input over its range, −1 if it falls"""
        f = vnames.index(d['feature']); lo, hi = float(kn[0, f]), float(kn[-1, f]); xs = np.linspace(lo - .3 * (hi - lo + 1e-9), hi + .3 * (hi - lo + 1e-9), 25)
        g = sum(c * (xs if k == 'lin' else np.maximum(0, xs - t) if k == 'gt' else np.maximum(0, t - xs)) for c, k, t in d['terms'])
        return 1 if np.polyfit(xs, g, 1)[0] >= 0 else -1
    for hi_, hd in enumerate(heads):
        selp = head_sel_phrase(hd); hd['selects'] = selp; facts = []
        for o_ in range(16):
            v = O[:, hi_, o_]; ins = hd['neurons'][o_]
            Oa = O.copy(); Oa[:, hi_, o_] = v.mean(); La = logits(Oa); dL = (Lm_ - La)
            drop = a0 - float((La.argmax(1) == ref).mean())
            means = [float(v[ytrue == c].mean()) for c in range(10)]
            auc = [float(roc_auc_score(ytrue == c, v)) if 0 < (ytrue == c).sum() < len(v) else 0.5 for c in range(10)]
            eff = [float(dL[:, c].mean()) for c in range(10)]; mag = np.abs(dL).mean(0)
            order = np.argsort(-np.abs(eff))[:3]
            role = '; '.join(f'{"raises" if eff[c] > 0 else "lowers"} the {CLASSES[c]} score ({eff[c]:+.2f} on average)' for c in order if abs(eff[c]) > .02) or 'hardly moves any class score'
            tot = sum(d['importance'] for d in ins) or 1.0; parts = []
            for d in ins[:3]:
                hi_p, lo_p = PHRASE.get(d['feature'], (f'high {d["feature"]}', f'low {d["feature"]}'))
                parts.append(f'{"rises" if direction(d) > 0 else "falls"} with {d["feature"]} — up for {hi_p if direction(d) > 0 else lo_p} ({100 * d["importance"] / tot:.0f} %)')
            top = ins[0] if ins else None
            title = (f'{(PHRASE.get(top["feature"], ("high " + top["feature"], "low " + top["feature"]))[0 if direction(top) > 0 else 1]) if top else "constant"}'
                     + (f', {PHRASE.get(ins[1]["feature"], ("high " + ins[1]["feature"], "low " + ins[1]["feature"]))[0 if direction(ins[1]) > 0 else 1]}' if len(ins) > 1 else ''))
            big = int(np.argmax(means)); sep = int(np.argmax(np.abs(np.array(auc) - .5)))
            # particle groups: the neuron's top 2 inputs cut at their kink → up to 4 groups of particles (real particles only)
            groups = []
            if m['kind'] == 'heads' and ins:
                cuts = []
                for d in ins[:2]:
                    fi = vnames.index(d['feature']); kt = [t for c_, k_, t in d['terms'] if k_ in ('gt', 'lt')]
                    cuts.append((d['feature'], fi, kt[0] if kt else float(np.median(XR[:, fi]))))
                fval = TR @ Wv[hi_][:6 * nv, o_]; awt = AR[hi_ // 8][:, hi_ % 8]; contrib = awt * fval; tot_c = np.abs(contrib).sum() or 1.0
                ms = [[(f'{n_} < {t_:.3g}', XR[:, fi] < t_), (f'{n_} ≥ {t_:.3g}', XR[:, fi] >= t_)] for n_, fi, t_ in cuts]
                combos = [(a[0] + ' and ' + b[0], a[1] & b[1]) for a in ms[0] for b in ms[1]] if len(ms) == 2 else ms[0]
                for name, g in combos:
                    ng = int(g.sum())
                    if ng == 0: continue
                    groups.append(dict(name=name, share=ng / len(g), mean_f=float(fval[g].mean()), mean_w=float(awt[g].mean() * nmean), contrib=float(contrib[g].sum() / tot_c),
                                       kinds={k_: float(v_[g].mean()) for k_, v_ in KR.items()}, hard=float(ZR[g].mean()),
                                       by_class=[float(contrib[g & (YR == c)].sum() / max((ytrue == c).sum(), 1)) for c in range(10)]))
            facts.append(dict(title=title, importance=None, strength=float(mag.mean()), drop=drop, groups=groups,
                              measures=f'Sums, over the particles this head weights ({selp}), a formula that ' + '; '.join(parts) + '.',
                              role=f'This neuron {role} (its average contribution, against the neuron held at its mean); without it, agreement with ParT changes by {-100 * drop:+.2f} pt.',
                              scale=f'largest for {CLASSES[big]} jets; separates {CLASSES[sep]} jets best (AUC {max(auc[sep], 1 - auc[sep]):.2f})',
                              means=means, auc=auc, effect=eff, ifs=if_lines(ins)))
        hd['facts'] = facts
        # the head as a whole: remove it (all 16 values at their means) → class-score changes; what its neurons read; selection by class
        Oa = O.copy(); Oa[:, hi_] = O[:, hi_].mean(0); La = logits(Oa); dLh = Lm_ - La; eh = [float(dLh[:, c].mean()) for c in range(10)]
        agg = {}
        for ins_ in hd['neurons']:
            for d in ins_: agg[d['feature']] = agg.get(d['feature'], 0.0) + d['importance']
        reads = sorted(agg, key=agg.get, reverse=True)[:5]; tot_r = sum(agg.values()) or 1.0
        b_ = hi_ // 8; alh = A[b_, :, hi_ % 8, 1:]; effn = np.exp(-(alh * np.log(alh + 1e-12)).sum(1)); selfw = A[b_, :, hi_ % 8, 0]
        by_cls = [dict(cls=CLASSES[c], eff=float(effn[ytrue == c].mean()), self=float(selfw[ytrue == c].mean())) for c in range(10)]
        oh = np.argsort(-np.abs(eh))[:3]
        hd['summary'] = dict(title=f'reads {", ".join(reads[:2])} of {selp}', reads=[(r, agg[r] / tot_r) for r in reads], effect=eh, by_class=by_cls,
                             serves='; '.join(f'{"raises" if eh[c] > 0 else "lowers"} the {CLASSES[c]} score ({eh[c]:+.2f})' for c in oh))
        log(f'  block {hd["block"]} head {hd["head"]}: per-neuron facts done ({sum(f["importance"] == "major" for f in facts)} major)')
    for hi_, hd in enumerate(heads):                                        # ParT's fixed arithmetic after this head, in numbers
        b_ = hi_ // 8; blk = model.cls_blocks[b_]; h_ = hi_ % 8
        Wo = blk.attn.out_proj.weight.detach().cpu().numpy()[:, 16 * h_:16 * h_ + 16]; ca = float(blk.c_attn.detach().cpu().numpy()[h_])
        colnorm = np.linalg.norm(Wo, axis=0); hd['combine'] = dict(c_attn=ca, out_norm=[float(v) for v in colnorm], out_total=float(np.linalg.norm(Wo)),
                                                                   heads_total=[float(np.linalg.norm(blk.attn.out_proj.weight.detach().cpu().numpy()[:, 16 * k:16 * k + 16])) for k in range(8)],
                                                                   heads_c=[float(v) for v in blk.c_attn.detach().cpu().numpy()])
    allst = np.array([f['strength'] for hd in heads for f in hd['facts']]); q80, q50 = np.quantile(allst, .8), np.quantile(allst, .5)
    for hd in heads:                                                        # importance: how much the neuron moves the class scores, ranked over all 256
        for f in hd['facts']: f['importance'] = 'major' if f['strength'] >= q80 else 'moderate' if f['strength'] >= q50 else 'minor'
    # example jets: n_ex per ParT class (the smallest multiplicities ≥ 8 first, then random), with every head's weights and the score decomposition
    rng = np.random.default_rng(0); ex = []
    for c in range(10):
        cand = np.flatnonzero((ref == c) & (npart >= 8)); pick = list(cand[np.argsort(npart[cand])[:n_ex // 2]]) + list(rng.choice(cand, n_ex - n_ex // 2, replace=False))
        for i in pick:
            ps = np.flatnonzero(ok[i]); parts = [dict(eta=round(float(F0[i, p, 5]), 4), phi=round(float(F0[i, p, 6]), 4), z=float(np.exp(F0[i, p, 2])), dr=float(F0[i, p, 4]), type=TYPES[int(np.argmax(F0[i, p, 8:13]))], q=int(F0[i, p, 7]), d0s=float(d0s[i, p])) for p in ps]
            hs = []
            for h in range(16):
                b = h // 8; w = A[b, i, h % 8]; e = dict(self=float(w[0]), w=[float(w[1 + p]) for p in ps])
                if S is not None and b == 0:
                    e['s'] = [float(S[i, p, h]) for p in ps]; W_s = m['Ws'][:, h]; tops = []
                    for p in ps:
                        Fs = (F[i, p] - m['mus']) / m['sds']; phi = np.concatenate([[1], Fs] + [np.maximum(0, F[i, p] - k) / m['sds'] for k in m['kns']]); con = phi * W_s; byf = np.zeros(nf)
                        for j in range(1, len(con)): byf[(j - 1) % nf] += con[j]
                        tj = np.argsort(-np.abs(byf))[:5]; tops.append([[names[f], round(float(byf[f]), 2)] for f in tj])
                    e['top'] = tops
                hs.append(e)
            sm = np.exp(Lm[i] - Lm[i].max()); sm /= sm.sum(); sp = np.exp(L[i] - L[i].max()); sp /= sp.sum()
            xin = np.round(F[i, ps][:, :nv], 5).tolist() if m['kind'] == 'heads' else None
            ex.append(dict(X=xin, part=int(ref[i]), truth=int(J['y'][rows[i]]), model=int(pred[i]), p_model=sm.round(3).tolist(), p_part=sp.round(3).tolist(), n=int(npart[i]), particles=parts, heads=hs))
    model_js = dict(W=np.round(m['W'], 6).tolist(), kn=np.round(m['kn'][:, :nv], 6).tolist(), nv=nv, names=vnames) if m['kind'] == 'heads' else None
    hists = {vnames[f]: HIST[f] for f in range(nv)} if m['kind'] == 'heads' else {}
    for hd in heads:
        for ins_ in hd['neurons']:
            for d in ins_:
                for st in d.get('stats', []): st.pop('hist', None)
    return dict(tag=tag, agreement=a0, heads=heads, examples=ex, model_js=model_js, hists=hists)


def svg_lines(x, Y, w=250, h=100, color='#1f4e79', op=.45):
    xs, Y = np.array(x), np.atleast_2d(np.array(Y)); lo, hi = Y.min(), Y.max(); hi = hi if hi > lo else lo + 1
    X = lambda v: 30 + (w - 40) * (v - xs[0]) / (xs[-1] - xs[0] + 1e-12); Yp = lambda v: h - 18 - (h - 28) * (v - lo) / (hi - lo)
    lines = ''.join(f'<polyline points="{" ".join(f"{X(a):.1f},{Yp(b):.1f}" for a, b in zip(xs, row))}" fill="none" stroke="{color}" stroke-opacity="{op}" stroke-width="1.4"/>' for row in Y)
    return (f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px"><line x1="30" x2="{w - 10}" y1="{Yp(0):.1f}" y2="{Yp(0):.1f}" stroke="#ccc"/>{lines}'
            f'<text x="30" y="{h - 4}" font-size="10" fill="#667">{xs[0]:.2g}</text><text x="{w - 10}" y="{h - 4}" font-size="10" text-anchor="end" fill="#667">{xs[-1]:.2g}</text></svg>')


def svg_profile(p, w=250, h=90):
    e, wt = p['edges'], [v if v is not None else 0 for v in p['weight']]; mx = max(wt) if max(wt) > 0 else 1; n = len(wt); bw = (w - 40) / n
    bars = ''.join(f'<rect x="{30 + i * bw:.1f}" y="{h - 18 - (h - 30) * v / mx:.1f}" width="{bw - 2:.1f}" height="{(h - 30) * v / mx:.1f}" fill="#1f4e79" opacity=".8"/>' for i, v in enumerate(wt))
    lab = lambda v: '∞' if v is None else f'{v:.2g}'
    return (f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px">{bars}<line x1="30" x2="{w - 10}" y1="{h - 18 - (h - 30) / mx:.1f}" y2="{h - 18 - (h - 30) / mx:.1f}" stroke="#c05621" stroke-dasharray="3 3"/>'
            f'<text x="30" y="{h - 4}" font-size="10" fill="#667">{lab(e[0])}</text><text x="{w - 10}" y="{h - 4}" font-size="10" text-anchor="end" fill="#667">{lab(e[-1])}</text></svg>')


def page(tag, outdir, device='mps', log=print):
    an = analyze(tag, device=device, log=log); title, kept, formula = INFO[tag]; outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    te = next((json.loads(p.read_text()) for p in (OUT / f'{tag}_metrics_full_test.json', OUT / f'{tag}_model_metrics_full_test.json') if p.exists()), None)
    base = json.loads((RESULTS / 'all_plus' / 'full' / 'metrics.json').read_text()); bm = next(x for x in base['models'] if x.get('terms') == 1083); pct = lambda x: f'{100 * x:.2f}%'
    joint = tag.startswith('S10')
    if te:
        cols = list(te['rej']); t = '<table><tr><th>model</th><th class="num">same class as ParT</th><th class="num">accuracy</th><th class="num">AUC</th>' + ''.join(f'<th class="num">Rej<sub>{int(te["rej"][c]["eff"] * 1000) / 10:g}%</sub> {c}</th>' for c in cols) + '</tr>'
        for name, mm, ag in (('ParT', te['part'], None), ('jet-level formula, all_plus (1083 terms)', bm['paper'], bm['same_as_network']), (title, te, te['agreement'])):
            t += f'<tr><td>{html.escape(name)}</td><td class="num">{"—" if ag is None else pct(ag)}</td><td class="num">{mm["accuracy"]:.4f}</td><td class="num">{mm["auc"]:.4f}</td>' + ''.join(f'<td class="num">{mm["rej"][c]["rej"]:.0f}</td>' for c in cols) + '</tr>'
        pc = '<table><tr><th>class</th><th class="num">same class as ParT</th><th class="num">accuracy</th></tr>' + ''.join(f'<tr><td>{CLASSES[p["c"]]}</td><td class="num">{pct(p["agreement"])}</td><td class="num">{pct(p["accuracy"])}</td></tr>' for p in te['per_class']) + '</table>'
        metrics_html = f'<h2>The ParT paper’s metrics (2,000,000 test jets)</h2><div class="card">{t}</table></div><h2>Per class</h2><div class="card">{pc}</div>'
        kpi = f'<span class="kpi">same class as ParT (test)<b>{pct(te["agreement"])}</b></span><span class="kpi">accuracy<b>{te["accuracy"]:.4f}</b><span class="cnt">ParT {te["part"]["accuracy"]:.4f}</span></span><span class="kpi">AUC<b>{te["auc"]:.4f}</b><span class="cnt">ParT {te["part"]["auc"]:.4f}</span></span>'
    else:
        metrics_html = ''; kpi = f'<span class="kpi">same class as ParT<b>{pct(an["agreement"])}</b><span class="cnt">balanced validation sample, 20,000 jets (test pending)</span></span>'
    selmode = lambda hd: ('the score formula, per jet softmax with the class token' if joint and hd['block'] == 1 else 'a plain average over the particles' if (joint or tag == 'S8_uniform2') and hd['block'] == 2 else 'ParT’s attention weights (not a formula)')
    hb = ''.join(f'<button class="tab hb{" on" if i == 0 else ""}" title="{html.escape(hd["summary"]["title"])}" onclick="setHead({i})">block {hd["block"]} · head {hd["head"]} <small>−{100 * hd["drop"]:.1f}</small></button>' for i, hd in enumerate(an['heads']))
    jb = ''.join(f'<span class="btn jb{" on" if i == 0 else ""}" onclick="setJet({i})" style="border-color:{"#2f855a" if e["model"] == e["part"] else "#c05621"}">{CLASSES[e["part"]]} <small>{e["n"]}p</small></span>' for i, e in enumerate(an['examples']))
    panels = ''
    def hist_svg(hs, thr, w=260, h=70):
        e, c = hs['edges'], hs['counts']; mx = max(c) or 1; n = len(c); bw = (w - 20) / n
        X = lambda v: 10 + (w - 20) * (v - e[0]) / (e[-1] - e[0] + 1e-12)
        bars = ''.join(f'<rect x="{10 + i * bw:.1f}" y="{h - 14 - (h - 22) * v / mx:.1f}" width="{bw - 1:.1f}" height="{(h - 22) * v / mx:.1f}" fill="#c9d4e3"/>' for i, v in enumerate(c))
        mark = '' if thr is None else f'<line x1="{X(thr):.1f}" x2="{X(thr):.1f}" y1="4" y2="{h - 12}" stroke="#c05621" stroke-width="2"/><text x="{X(thr) + 3:.1f}" y="12" font-size="10" fill="#c05621">{thr:.3g}</text>'
        return f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px">{bars}{mark}<text x="10" y="{h - 2}" font-size="9" fill="#667">{e[0]:.3g}</text><text x="{w - 10}" y="{h - 2}" font-size="9" text-anchor="end" fill="#667">{e[-1]:.3g}</text></svg>'
    def terms_html(inputs, share=.99, max_details=12):
        tot = sum(d['importance'] for d in inputs) or 1.0; acc, out, plain = 0.0, [], []
        for d in inputs:
            if acc >= share * tot and len(out) >= 3: break
            acc += d['importance']
            for (c, k, t), st in zip(d['terms'], d['stats'] if d.get('stats') else [None] * len(d['terms'])):
                if len(out) >= max_details:
                    v = f'particle["{d["feature"]}"]'; plain.append(f'add {c:+.4g} · {v}' if k == 'lin' else f'if {v} > {t:.4g}:  add {c:+.4g} · ({v} − {t:.4g})' if k == 'gt' else f'if {v} < {t:.4g}:  add {c:+.4g} · ({t:.4g} − {v})'); continue
                v = f'particle["{d["feature"]}"]'
                line = f'add {c:+.4g} · {v}' if k == 'lin' else f'if {v} > {t:.4g}:  add {c:+.4g} · ({v} − {t:.4g})' if k == 'gt' else f'if {v} < {t:.4g}:  add {c:+.4g} · ({t:.4g} − {v})'
                if st is None: out.append(f'<details class="xterm"><summary><code>{html.escape(line)}</code> <span class="cnt">±{d["importance"]:.3g}</span></summary></details>'); continue
                kinds = ', '.join(f'{k2} {100 * v2:.0f} %' for k2, v2 in sorted(st['kinds'].items(), key=lambda q: -q[1]) if v2 > .05)
                byc = sorted(enumerate(st['by_class']), key=lambda q: -abs(q[1]))[:4]; mxc = max(abs(v2) for _, v2 in byc) or 1
                cls = ''.join(f'<tr><td>{CLASSES[ci]}</td><td style="width:120px"><div class="bar" style="width:{100 * abs(v2) / mxc:.0f}px;background:{"#2f855a" if v2 >= 0 else "#c05621"}"></div></td><td class="num">{v2:+.3g}</td></tr>' for ci, v2 in byc)
                when = 'always (a linear term)' if k == 'lin' else f'for {100 * st["fires"]:.0f} % of all particles'
                out.append(f'''<details class="xterm"><summary><code>{html.escape(line)}</code> <span class="cnt">typical contribution ±{d["importance"]:.3g}</span></summary>
<div class="grid" style="margin:6px 0 4px 10px"><div><div><b>{html.escape(d["feature"])}</b>: {html.escape(WORDS.get(d["feature"], ""))} <span class="cnt">({html.escape(DEFS.get(d["feature"], ""))})</span></div>
<div style="margin-top:4px">Fires {when}{"" if k == "lin" else f"; those are {kinds}; hard (pT share > 5 %) {100 * st['hard']:.0f} %"}. The amount added grows with the distance from the threshold.</div></div>
<div><div class="cnt">distribution of this input over the particles (threshold in orange)</div><div class="hh" data-f="{html.escape(d["feature"])}" data-thr="{'' if st['thr'] is None else st['thr']}"></div></div>
<div><div class="cnt">what this statement adds to the neuron, per jet, by true class (α-weighted)</div><table class="bars">{cls}</table></div></div></details>''')
        if plain: out.append('<details class="xterm"><summary><span class="cnt">' + str(len(plain)) + ' further statements (smaller)</span></summary><pre>' + html.escape(chr(10).join(plain)) + '</pre></details>')
        rest = max(0, sum(len(d['terms']) for d in inputs) - sum(len(d['terms']) for d in inputs if acc >= 0) + 0); out.append('')
        return ''.join(out)
    def groups_html(gs):
        if not gs: return '<p class="cnt">no groups</p>'
        top_cls = lambda g: ', '.join(f'{CLASSES[c]} {v:+.2f}' for c, v in sorted(enumerate(g['by_class']), key=lambda t: -abs(t[1]))[:3])
        words = ''.join(f'<li><b>{html.escape(g["name"])}</b>: {html.escape(group_words(g, CLASSES))}</li>' for g in gs)
        rows = ''.join(f'<tr><td>{html.escape(g["name"])}</td><td class="num">{100 * g["share"]:.1f} %</td><td class="num">{g["mean_f"]:+.3g}</td><td class="num">{g["mean_w"]:.2f}</td>'
                       f'<td class="num">{100 * g["contrib"]:+.1f} %</td><td>{", ".join(f"{k} {100 * v:.0f} %" for k, v in sorted(g["kinds"].items(), key=lambda t: -t[1]) if v > .05)}; hard {100 * g["hard"]:.0f} %</td><td>{top_cls(g)}</td></tr>' for g in gs)
        return ('<table><tr><th>particles with</th><th class="num">share of particles</th><th class="num">mean f</th><th class="num">mean weight ×n</th><th class="num">share of the neuron</th><th>what they are</th><th>jets where they add most (per jet)</th></tr>'
                + rows + '</table><ul style="margin:8px 0 4px 18px">' + words + '</ul><p class="cnt">mean weight ×n: the head’s average weight on these particles times the jet’s multiplicity (1 = an average particle). Share of the neuron: their part of Σ α·f over all jets. Hard: pT share above 5 %.</p>')
    def bars(vals, lab, fmt, center=None):
        mx = max(abs(x - (center or 0)) for x in vals) or 1
        return '<table class="bars">' + ''.join(f'<tr><td>{CLASSES[c]}</td><td style="width:150px"><div class="bar" style="width:{130 * abs(v - (center or 0)) / mx:.0f}px;background:{"#2f855a" if v - (center or 0) >= 0 else "#c05621"}"></div></td><td class="num">{fmt(v)}</td></tr>' for c, v in enumerate(vals)) + f'</table><div class="cnt">{lab}</div>'
    imp_col = {'major': '#1d2433', 'moderate': '#98a2b3', 'minor': '#e4e7ec'}
    for i, hd in enumerate(an['heads']):
        sel = hd['selection']; prof = ''.join(f'<div><div class="cnt">weight vs {html.escape(k)} <small>(×average particle; dashed = 1)</small></div>{svg_profile(v)}</div>' for k, v in sel.items() if isinstance(v, dict))
        flags = ' · '.join(f'{k} ×{v:.2f}' for k, v in sel.items() if not isinstance(v, dict) and v is not None)
        rule = ''
        if hd['rule']:
            rb = '<table class="bars">' + ''.join(f'<tr><td>{html.escape(r["feature"])}</td><td style="width:180px"><div class="bar" style="width:{160 * r["importance"] / hd["rule"]["inputs"][0]["importance"]:.0f}px"></div></td></tr>' for r in hd['rule']['inputs']) + '</table>'
            rcv = ''.join(f'<div><div class="cnt">score vs {html.escape(k)} <small>(net, other inputs fixed)</small></div>{svg_lines(v["x"], v["y"], color="#c05621", op=.9)}</div>' for k, v in list(hd['rule']['curves'].items())[:6])
            rule = f'<h3>The rule: score formula of this head</h3><div class="card"><div class="cnt">score(particle) = {hd["rule"]["intercept"]:+.2f} + Σ terms; the class token’s own score is 0; weights = softmax over the jet. Inputs by weight (coefficient × spread):</div>{rb}<div class="grid">{rcv}</div></div>'
        fb = '<table class="bars">' + ''.join(f'<tr><td>{html.escape(f["feature"])}</td><td style="width:180px"><div class="bar" style="width:{160 * f["importance"] / hd["features"][0]["importance"]:.0f}px"></div></td></tr>' for f in hd['features']) + '</table>'
        cv = ''.join(f'<div><div class="cnt">16 neurons vs {html.escape(k)}</div>{svg_lines(v["x"], v["y"])}</div>' for k, v in list(hd['curves'].items())[:4])
        imp_col = {'major': '#1d2433', 'moderate': '#98a2b3', 'minor': '#e4e7ec'}
        nb = ''.join(f'<span class="nbtn{" on" if j == 0 else ""}" style="border-left:4px solid {imp_col[hd["facts"][j]["importance"]]}" title="{html.escape(hd["facts"][j]["title"])}" onclick="showNeu({i},{j})">{j + 1}</span>' for j in range(16))
        def bars(vals, lab, fmt, center=None):
            mx = max(abs(x - (center or 0)) for x in vals) or 1
            return '<table class="bars">' + ''.join(f'<tr><td>{CLASSES[c]}</td><td style="width:150px"><div class="bar" style="width:{130 * abs(v - (center or 0)) / mx:.0f}px;background:{"#2f855a" if v - (center or 0) >= 0 else "#c05621"}"></div></td><td class="num">{fmt(v)}</td></tr>' for c, v in enumerate(vals)) + f'</table><div class="cnt">{lab}</div>'
        np_ = ''.join(f'''<div class="neu{" on" if j == 0 else ""}" id="n{i}_{j}">
<h3 style="margin:8px 0 2px">Neuron {j + 1}: {html.escape(fa["title"])} <span class="pill" style="background:{imp_col[fa["importance"]]};color:{"#fff" if fa["importance"] == "major" else "#1d2433"}">{fa["importance"]}</span></h3>
<p style="margin:4px 0"><b>What it measures.</b> {html.escape(fa["measures"])}</p><p style="margin:4px 0"><b>Role.</b> {html.escape(fa["role"])} <span class="cnt">Scale: {html.escape(fa["scale"])}.</span></p>
<div class="grid"><div><b>As if-statements</b> <span class="cnt">(per particle i; the neuron = Σ_i α_i · (sum of these) + c·α_cls + b)</span><pre>{html.escape(chr(10).join(fa["ifs"]))}</pre></div>
<div><b>As a formula</b><pre>{html.escape(neuron_snippet(tag, hd["block"], hd["head"], j, hd["neurons"][j]))}</pre></div></div>
<div class="grid"><div><b>Mean value by true class</b>{bars(fa["means"], "the neuron’s mean on jets of each true class", lambda v: f"{v:.2f}", center=float(np.mean(fa["means"])))}</div>
<div><b>Separation (AUC vs the rest)</b>{bars(fa["auc"], "0.5 = no separation; above: higher for that class", lambda v: f"{v:.2f}", center=0.5)}</div>
<div><b>Effect on the class scores</b>{bars(fa["effect"], "average change of each class score caused by this neuron (vs its mean)", lambda v: f"{v:+.2f}")}</div></div></div>''' for j, fa in enumerate(hd['facts']))
        dd = ''.join(f'''<details class="nd"><summary><b>Neuron {j + 1}</b> — {html.escape(fa["title"])} <span class="pill" style="background:{imp_col[fa["importance"]]};color:{"#fff" if fa["importance"] == "major" else "#1d2433"}">{fa["importance"]}</span></summary>
<p style="margin:6px 0"><b>What it measures.</b> {html.escape(fa["measures"])}</p><p style="margin:4px 0"><b>Role.</b> {html.escape(fa["role"])} <span class="cnt">Scale: {html.escape(fa["scale"])}.</span></p>
<details open><summary><b>If-statements</b> <span class="cnt">(applied to each particle of the jet in turn; the neuron = Σ over the particles of α<sub>i</sub> · f(particle i) + c·α<sub>cls</sub> + b; open a statement for its details)</span></summary>{terms_html(hd["neurons"][j])}</details>
<details><summary><b>Formula</b></summary><pre>{html.escape(neuron_snippet(tag, hd["block"], hd["head"], j, hd["neurons"][j]))}</pre></details>
<details><summary><b>Python code</b> <span class="cnt">(complete, every term)</span></summary><pre>{html.escape(neuron_python(hd["block"], hd["head"], j, hd["neurons"][j], *hd["cb"][j]))}</pre></details>
<details><summary><b>Particle groups</b> <span class="cnt">(this neuron as a particle tagger: its top inputs cut at their thresholds)</span></summary>{groups_html(fa["groups"])}</details>
<details><summary><b>By class</b></summary><div class="grid"><div><b>Mean value by true class</b>{bars(fa["means"], "the neuron’s mean on jets of each true class", lambda v: f"{v:.2f}", center=float(np.mean(fa["means"])))}</div>
<div><b>Separation (AUC vs the rest)</b>{bars(fa["auc"], "0.5 = no separation", lambda v: f"{v:.2f}", center=0.5)}</div><div><b>Effect on the class scores</b>{bars(fa["effect"], "average change of each class score caused by this neuron", lambda v: f"{v:+.2f}")}</div></div></details>
</details>''' for j, fa in enumerate(hd['facts']))
        panels += f'''<div class="head{" on" if i == 0 else ""}" id="h{i}"><h2>Block {hd["block"]}, head {hd["head"]} <span class="cnt">— removing it: −{100 * hd["drop"]:.2f} pt agreement · class token’s own share {hd["self_weight"]:.2f} · {hd["eff_particles"]:.1f} effective particles · selection: {selmode(hd)}</span></h2>
<div class="card"><b>What this head does:</b> {html.escape(hd["summary"]["title"])}.
<p style="margin:6px 0"><b>Selects</b> {html.escape(hd["selects"])} — about {hd["eff_particles"]:.0f} effective particles per jet, the class token keeping {100 * hd["self_weight"]:.0f} % for itself.<br>
<b>Reads</b> (summed over its 16 neurons): {html.escape(", ".join(f"{r} ({100 * w:.0f} %)" for r, w in hd["summary"]["reads"]))}.<br>
<b>Serves</b> — this head {html.escape(hd["summary"]["serves"])} (its average contribution); removing it costs {100 * hd["drop"]:.2f} pt of agreement with ParT.</p>
<div class="grid"><div><b>Effect of the head on the class scores</b>{bars(hd["summary"]["effect"], "average change of each class score caused by this head", lambda v: f"{v:+.2f}")}</div>
<div><b>Its selection by true class</b><table><tr><th>class</th><th class="num">effective particles</th><th class="num">class-token share</th></tr>{"".join(f'<tr><td>{x["cls"]}</td><td class="num">{x["eff"]:.1f}</td><td class="num">{100 * x["self"]:.1f} %</td></tr>' for x in hd["summary"]["by_class"])}</table></div></div></div>
<h3>The 16 neurons of this head</h3><div class="card"><p style="margin:0 0 8px"><b>How these 16 neurons are added together</b> (ParT’s fixed arithmetic, not fitted): the 16 values o<sub>1</sub>…o<sub>16</sub> of this head are multiplied by this head’s 16 columns of the block’s out-projection matrix W<sub>o</sub> (128 × 128) and summed into 128 numbers, u = Σ<sub>m</sub> o<sub>m</sub> · W<sub>o</sub>[:, m], then scaled by this head’s factor c = {hd["combine"]["c_attn"]:.3f}. Weight of each neuron in that sum (‖W<sub>o</sub>[:, m]‖): {", ".join(f"n{m + 1} {v:.2f}" for m, v in enumerate(hd["combine"]["out_norm"]))}.</p>{dd}
<p class="cnt" style="margin-top:8px"><b>How the 8 heads of block {hd["block"]} are added together:</b> the 8 heads’ 128-number contributions are summed (u = Σ<sub>h</sub> c<sub>h</sub> · Σ<sub>m</sub> o<sub>hm</sub> W<sub>o</sub>[:, hm]), LayerNorm-ed, and <b>added to the class token</b>; then the block’s MLP (LayerNorm → 512 GELU → LayerNorm → 128) is added on top. Size of each head in the sum (c<sub>h</sub> · ‖its W<sub>o</sub> columns‖): {", ".join(f"h{k + 1} {c * n:.2f}" for k, (c, n) in enumerate(zip(hd["combine"]["heads_c"], hd["combine"]["heads_total"])))}. {"Block 2 adds its result to block 1’s class token; the final LayerNorm then gives the 128 neurons and the last layer the class scores." if hd["block"] == 2 else "Block 2 then reads the same particles again with this updated class token."}</p></div>
{rule}<h3>The rule applied to a jet</h3><div class="slot"></div><h3>What the selection does on average</h3><div class="card"><div class="grid">{prof}</div><div class="cnt">{flags}</div></div>
<h3>What it sums over the selected particles: the 16 value neurons</h3><div class="card"><p style="margin:0 0 8px"><b>How to read a neuron:</b> neuron = Σ over <b>every particle i of the jet</b> of α<sub>i</sub> · f(particle i) + c · α<sub>cls</sub> + b. f is the function written below, evaluated on that particle’s inputs; α<sub>i</sub> is this head’s weight for that particle in this jet (the weights of all particles and the class token sum to 1, so it is a weighted average). All 16 neurons of a head share the same weights α and differ in f: 16 properties averaged over the same selection.</p><span class="cnt">inputs by total weight:</span>{fb}<div class="grid">{cv}</div><div style="margin-top:8px"><span class="cnt">neuron:</span> {nb}</div>{np_}</div></div>'''
    js = """const EX=__EX__, CL=__CL__, MJ=__MJ__, HS=__HS__; let H=0, JJ=0;
function histSVG(hs,thr){const w=260,h=70,e=hs.edges,c=hs.counts,mx=Math.max(...c)||1,n=c.length,bw=(w-20)/n,X=v=>10+(w-20)*(v-e[0])/(e[e.length-1]-e[0]+1e-12);
 let g='';c.forEach((v,i)=>{g+='<rect x="'+(10+i*bw).toFixed(1)+'" y="'+(h-14-(h-22)*v/mx).toFixed(1)+'" width="'+(bw-1).toFixed(1)+'" height="'+((h-22)*v/mx).toFixed(1)+'" fill="#c9d4e3"/>';});
 if(thr!==''){const t=parseFloat(thr);g+='<line x1="'+X(t).toFixed(1)+'" x2="'+X(t).toFixed(1)+'" y1="4" y2="'+(h-12)+'" stroke="#c05621" stroke-width="2"/><text x="'+(X(t)+3).toFixed(1)+'" y="12" font-size="10" fill="#c05621">'+t.toPrecision(3)+'</text>';}
 return '<svg viewBox="0 0 '+w+' '+h+'" style="width:100%;max-width:'+w+'px">'+g+'<text x="10" y="'+(h-2)+'" font-size="9" fill="#667">'+e[0].toPrecision(3)+'</text><text x="'+(w-10)+'" y="'+(h-2)+'" font-size="9" text-anchor="end" fill="#667">'+e[e.length-1].toPrecision(3)+'</text></svg>';}
document.addEventListener('toggle',ev=>{const d=ev.target; if(!d.classList||!d.classList.contains('xterm')||!d.open) return; d.querySelectorAll('.hh').forEach(x=>{if(!x.dataset.done){x.innerHTML=histSVG(HS[x.dataset.f],x.dataset.thr);x.dataset.done=1;}});},true);
function setHead(i){H=i;document.querySelectorAll('.head').forEach((e,k)=>e.classList.toggle('on',k==i));document.querySelectorAll('.hb').forEach((e,k)=>e.classList.toggle('on',k==i));document.querySelectorAll('.head')[i].querySelector('.slot').appendChild(document.getElementById('jetbox'));drawJet();}
function setJet(i){JJ=i;document.querySelectorAll('.jb').forEach((e,k)=>e.classList.toggle('on',k==i));drawJet();}
let NN=0; function setNN(j){NN=j;document.querySelectorAll('.nnb').forEach((e,k)=>e.classList.toggle('on',k==j));drawJet();}
function showNeu(i,j){ if(i==H){NN=j; drawJet();} for(let k=0;k<16;k++){document.getElementById('n'+i+'_'+k).classList.toggle('on',k==j);}document.querySelectorAll('#h'+i+' .nbtn').forEach((e,k)=>e.classList.toggle('on',k==j));}
function core(w){const o=w.map((v,k)=>[v,k]).sort((a,b)=>b[0]-a[0]); let acc=0, tot=w.reduce((a,b)=>a+b,0), S=new Set(); for(const [v,k] of o){if(acc>=0.9*tot) break; acc+=v; S.add(k);} return S;}
function jetSVG(e,h,S){const W=360,Hh=300,R=0.8, X=v=>W/2+v/R*(W/2-20), Y=v=>Hh/2-v/R*(Hh/2-20); const mx=Math.max(...h.w);
 let g='<svg viewBox="0 0 '+W+' '+Hh+'" style="width:100%;max-width:'+W+'px;background:#fbfcfe;border:1px solid #e4e7ec;border-radius:8px">';
 g+='<line x1="'+X(-R)+'" x2="'+X(R)+'" y1="'+Y(0)+'" y2="'+Y(0)+'" stroke="#eee"/><line y1="'+Y(-R)+'" y2="'+Y(R)+'" x1="'+X(0)+'" x2="'+X(0)+'" stroke="#eee"/>';
 [0.2,0.4,0.8].forEach(r=>{g+='<circle cx="'+X(0)+'" cy="'+Y(0)+'" r="'+(r/R*(W/2-20))+'" fill="none" stroke="#e4e7ec" stroke-dasharray="2 3"/><text x="'+(X(r)+2)+'" y="'+(Y(0)-2)+'" font-size="9" fill="#99a">ΔR '+r+'</text>';});
 const ord=e.particles.map((p,k)=>k).sort((a,b)=>h.w[a]-h.w[b]);
 ord.forEach(k=>{const p=e.particles[k], a=h.w[k]/mx, r=3+10*Math.sqrt(p.z); g+='<circle cx="'+X(p.eta)+'" cy="'+Y(p.phi)+'" r="'+r.toFixed(1)+'" fill="rgb('+Math.round(31+(192-31)*(1-a))+','+Math.round(78+(197-78)*(1-a))+','+Math.round(121+(220-121)*(1-a))+')" fill-opacity="'+(0.25+0.75*a).toFixed(2)+'" stroke="'+(S.has(k)?'#c05621':'none')+'" stroke-width="1.6" style="cursor:pointer" onclick="showP('+k+')"><title>particle '+(k+1)+': α = '+h.w[k].toFixed(3)+', pT share '+p.z.toFixed(3)+', '+p.type+'</title></circle>';});
 g+='<text x="8" y="'+(Hh-8)+'" font-size="10" fill="#667">Δη →   (Δφ ↑) · size: pT share · colour: α in this head (dark = high) · orange ring: carries 90 % of the weight</text></svg>'; return g;}
function drawJet(){const e=EX[JJ], h=e.heads[H], hasS=h.s!==undefined; let mx=Math.max(h.self,...h.w); const S=core(h.w);
 let s='<div class="cnt">ParT: <b>'+CL[e.part]+'</b> ('+(100*e.p_part[e.part]).toFixed(0)+'%) · this model: <b>'+CL[e.model]+'</b> ('+(100*e.p_model[e.model]).toFixed(0)+'%) · truth '+CL[e.truth]+' · '+e.n+' particles. Head b'+(H<8?1:2)+' h'+(H%8+1)+': '+(hasS?'score = formula(particle); ':'')+'weights = softmax over [class token, particles].'+(hasS?' Click a particle for what set its score.':'')+'</div>';
 s+='<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;margin:8px 0"><div>'+jetSVG(e,h,S)+'</div><div class="cnt" style="max-width:360px"><b>'+S.size+' of '+e.n+' particles</b> carry 90 % of this head’s weight on the particles (orange rings; highlighted rows). The class token keeps '+(100*h.self).toFixed(1)+' % for itself. Every particle is included in the sums; the others just count little. The weights α are recomputed for every jet — switch heads to see how differently they look at the same jet. Click a particle to see all its inputs.'+(MJ?'<br><br><b>How α is computed here:</b> by ParT. Its particle blocks give each particle an embedding e<sub>i</sub> (the particle in its jet context); then α<sub>i</sub> = exp(s<sub>i</sub>) / Σ<sub>j</sub> exp(s<sub>j</sub>) with s<sub>i</sub> = q · W<sub>k</sub>LN(e<sub>i</sub>) / 4, q = ParT’s query of the class token (block 1: the same for every jet; block 2: from block 1’s output), j over the class token and the particles. This is the one part of this model that is not a formula.':'')+'</div></div>';
 s+='<table><tr><th>#</th><th class="num">pT share</th><th class="num">ΔR</th><th>type</th><th class="num">charge</th><th class="num">|d0|/σ</th>'+(hasS?'<th class="num">score</th>':'')+'<th class="num">weight</th><th></th></tr>';
 s+='<tr class="cls"><td>class token</td><td></td><td></td><td></td><td></td><td></td>'+(hasS?'<td class="num">0.00</td>':'')+'<td class="num">'+h.self.toFixed(3)+'</td><td><span class="wbar" style="width:'+(120*h.self/mx)+'px"></span></td></tr>';
 e.particles.forEach((p,k)=>{s+='<tr class="p'+(S.has(k)?' core':'')+'" onclick="showP('+k+')"><td>'+(k+1)+'</td><td class="num">'+p.z.toFixed(3)+'</td><td class="num">'+p.dr.toFixed(2)+'</td><td>'+p.type+'</td><td class="num">'+(p.q>0?'+':'')+p.q+'</td><td class="num">'+p.d0s.toFixed(1)+'</td>'+(hasS?'<td class="num">'+h.s[k].toFixed(2)+'</td>':'')+'<td class="num">'+h.w[k].toFixed(3)+'</td><td><span class="wbar" style="width:'+(120*h.w[k]/mx)+'px"></span></td></tr>';});
 if(MJ&&e.X){const W=MJ.W[H], kn=MJ.kn, nv=MJ.nv, K=W.length;
  const fval=x=>{let v=0; for(let f=0;f<nv;f++){v+=W[f][NN]*x[f]; for(let k=0;k<5;k++){v+=W[nv+k*nv+f][NN]*Math.max(0,x[f]-kn[k][f]);}} return v;};
  const fv=e.X.map(fval); h.f=fv.map(v=>{const a=[];a[NN]=v;return a;}); h.c=W[K-2].slice(); h.b=W[K-1].slice();
  let tot=0; e.particles.forEach((p,k)=>{tot+=h.w[k]*h.f[k][NN];});
  let nb='<div style="margin-top:10px"><b>Worked example — value neuron:</b> '+[...Array(16).keys()].map(j=>'<span class="nbtn nnb'+(j==NN?' on':'')+'" onclick="setNN('+j+')">'+(j+1)+'</span>').join('')+'</div>';
  nb+='<table><tr><th>#</th><th class="num">weight α<sub>i</sub></th><th class="num">f(x<sub>i</sub>)</th><th class="num">α<sub>i</sub> · f(x<sub>i</sub>)</th></tr>';
  e.particles.forEach((p,k)=>{nb+='<tr><td>'+(k+1)+'</td><td class="num">'+h.w[k].toFixed(3)+'</td><td class="num">'+h.f[k][NN].toFixed(3)+'</td><td class="num">'+(h.w[k]*h.f[k][NN]).toFixed(3)+'</td></tr>';});
  const c=h.c[NN], b=h.b[NN]; nb+='</table><pre>neuron '+(NN+1)+' = Σ_i α_i · f(x_i)  +  c · α_cls  +  b\\n          = '+tot.toFixed(3)+'  +  '+c.toFixed(3)+' × '+h.self.toFixed(3)+'  +  '+b.toFixed(3)+'\\n          = '+(tot+c*h.self+b).toFixed(3)+'</pre><div class="cnt">f is the formula shown under “What it sums” below (neuron '+(NN+1)+'); every particle of the jet contributes its own f(x_i), weighted by this head’s α_i.</div>';
  s+='</table>'+nb; document.getElementById('jet').innerHTML=s;} else document.getElementById('jet').innerHTML=s+'</table>'; document.getElementById('contrib').innerHTML=hasS?'<span class="cnt">click a particle</span>':'<span class="cnt">this head’s weights are '+(hasS?'':'not from a formula on this page')+'</span>';}
function showP(k){const e=EX[JJ], h=e.heads[H]; document.querySelectorAll('#jet tr.p').forEach((r,i)=>r.classList.toggle('on',i==k));
 let t='<div class="cnt">particle '+(k+1)+' — weight α = '+h.w[k].toFixed(3)+' in this head.'+(MJ&&e.X?' Its inputs x_i (what f(x_i) is evaluated on):':'')+'</div>';
 if(MJ&&e.X){t+='<table>'+MJ.names.map((nm,f)=>'<tr><td>'+nm+'</td><td class="num">'+e.X[k][f].toPrecision(4)+'</td></tr>').join('')+'</table>';}
 if(h.top===undefined){document.getElementById('contrib').innerHTML=t; return;} document.querySelectorAll('#jet tr.p').forEach((r,i)=>r.classList.toggle('on',i==k));
 document.getElementById('contrib').innerHTML='<div class="cnt">particle '+(k+1)+': score '+h.s[k].toFixed(2)+' = intercept + the largest net contributions by input:</div><pre>'+h.top[k].map(t=>(t[1]>=0?'+':'')+t[1].toFixed(2)+'  '+t[0]).join('\\n')+'\\n  + smaller terms</pre>';}
setHead(0);""".replace('__EX__', json.dumps(an['examples'])).replace('__CL__', json.dumps(CLASSES)).replace('__MJ__', json.dumps(an['model_js'])).replace('__HS__', json.dumps(an.get('hists', {})))
    notation_rows = ''.join(f'<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(v)}</td></tr>' for k, v in NOTATION)
    defs_rows = ''.join(f'<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(WORDS.get(k, ""))}</td><td class="cnt">{html.escape(v)}</td></tr>' for k, v in DEFS.items()); m = load_model(tag); py = python_export(tag, m)
    if py: (outdir / f'{tag}_formulas.py').write_text(py)
    shutil.copy(OUT / f'{tag}_model.npz', outdir / f'{tag}_model.npz'); (outdir / 'analysis.json').write_text(json.dumps({k: v for k, v in an.items() if k not in ('examples', 'model_js', 'hists')}))
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>
<p class="cnt"><a href="../../index.html">← all setups</a> · <a href="../../research/index.html">research log</a></p><h1>{html.escape(title)}</h1>
<p class="cnt">ParT_full · JetClass · ParT’s class attention written per head. A head applies one rule to every jet: a score for each particle → attention weights by softmax (with the class token’s own share) → a sum over the particles of a formula of each particle’s physics (16 value neurons). ParT’s fixed arithmetic after the heads turns the 256 head outputs into the class scores.</p>
<div>{kpi}</div>
<div class="card"><div class="keep"><b>Kept of ParT:</b> {kept}<br><span class="cnt">{DOWN}</span></div><div class="form"><b>Formulas:</b> {formula}</div></div>
<h2>Notation</h2><div class="card"><table>{notation_rows}</table></div>
<h2>What the output is composed of</h2><div class="card"><pre>{html.escape(COMPOSE[joint])}</pre><p><b>How the per-particle tags are aggregated.</b> Each value neuron tags every particle (f); each head averages the tags with its weights (Σ<sub>i</sub> α<sub>i</sub> f(x<sub>i</sub>)): 16 numbers per head.
The particles themselves are not changed by the class blocks: both blocks read the same particles, only the class token is updated. In each class block the 8 heads × 16 are concatenated (128), mixed by a fixed linear map (out-projection), scaled per head, LayerNorm-ed and <b>added</b> to the class token;
a small MLP (LayerNorm → 512 GELU → LayerNorm → 128) is <b>added</b> on top; block 2 repeats this on block 1’s class token; a final LayerNorm gives the 128 neurons, and the last layer (a linear map 128 → 10) the class scores.</p>
<p class="cnt">Exact formulation with the fitted coefficients: <a href="{tag}_formulas.py">{tag}_formulas.py</a> (+ <a href="{tag}_model.npz">{tag}_model.npz</a>). Each head’s 16 value neurons are written out term by term in its tab.</p></div>
<div id="bar"><span class="cnt">head (and its cost when removed, in points of agreement):</span><br>{hb}</div>
<div id="jetbox"><div class="card"><div class="cnt">jet (6 per ParT class; green border = this model agrees with ParT on it):</div><div>{jb}</div></div><div class="card" id="jet"></div><div class="card" id="contrib"></div></div>
{panels}
{metrics_html}
<h2>Definitions of the per-particle inputs</h2><div class="card"><table><tr><th>input</th><th>in words</th><th>definition (math)</th></tr>{defs_rows}</table><p class="cnt">Every input describes <b>one particle i</b> of the jet — the particle a head is summing over. “Nearest” always means nearest <b>to particle i</b>, among the <b>other particles of the same jet</b>, by the angle ΔR = √(Δη² + Δφ²).</p><p class="cnt" style="margin-top:8px">{html.escape(PK_DEF)}</p><p class="cnt">Code: jetdistill/part/clsfit.py (particle_features), jetdistill/research/nbr.py (nbr_features, pk_features).</p></div>
<h2>Files</h2><div class="card"><a href="{tag}_formulas.py">{tag}_formulas.py</a> (exact formulation) · <a href="{tag}_model.npz">{tag}_model.npz</a> (coefficients, thresholds) · <a href="analysis.json">analysis.json</a> · code: branch part-features, jetdistill/research/{'joint.py' if joint else 'heads.py'}</div>
<script>{js}</script></main></body></html>"""
    (outdir / 'index.html').write_text(doc); log(f'page: {outdir / "index.html"} {len(doc) // 1024} kB')


if __name__ == '__main__':
    page(sys.argv[1], sys.argv[2])
