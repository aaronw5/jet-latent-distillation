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
from .heads_defs import DEFS, WORDS, PK_DEF, COMPOSE, NOTATION, python_export, neuron_snippet

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
.head{display:none}.head.on{display:block}#bar{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:6px 0 8px;margin-bottom:8px}.tab{border:none;background:none;padding:6px 10px;font:inherit;font-weight:600;color:var(--mut);cursor:pointer;border-bottom:3px solid transparent}.tab.on{color:var(--acc);border-bottom-color:var(--acc)}.tab small{font-weight:400}.nbtn{display:inline-block;border:1px solid #c9ced6;border-radius:6px;padding:1px 6px;margin:2px;cursor:pointer;background:#fff;font-size:12px}.nbtn.on{background:var(--acc);color:#fff;border-color:var(--acc)}.neu{display:none}.neu.on{display:block}
pre{font:12px/1.55 ui-monospace,Menlo,monospace;background:#f6f8fa;border:1px solid var(--line);border-radius:8px;padding:8px 10px;overflow-x:auto;white-space:pre-wrap;margin:4px 0}#contrib{min-height:60px}"""

INFO = {
    'S5': ('Per-head formulas, ParT’s selection', 'Both class blocks’ attention weights (which particles each head looks at, per jet) are ParT’s, computed from its particle blocks.',
           'Every head’s value — what it sums over the particles it attends to — is a formula of each particle’s own physics.'),
    'S8_uniform2': ('Per-head formulas, ParT’s block-1 selection', 'Class block 1’s attention weights are ParT’s (from its particle blocks); block 2 is a plain average over the particles.',
                    'Every head’s value is a formula of each particle’s own physics; block 2 needs no selection.'),
    'S10': ('All formulas', 'Nothing above the class attention: no particle blocks, no embeddings.',
            'Block 1’s selection (per head one score formula of a particle’s physics; per jet, softmax over the class token’s fixed score and the particles’ scores) and every head’s value are formulas; block 2 is a plain average.')}
INFO['S10b'] = INFO['S10']
DOWN = 'Kept in every model: ParT’s fixed arithmetic after the heads — out-projection, per-head scale, LayerNorms, the 128→512→128 MLP with residuals in both class blocks, the final LayerNorm and the last layer. Not fitted, no physics content: a smooth map from the 256 head outputs to the 10 class scores.'
TYPES = ['ch. hadron', 'n. hadron', 'photon', 'electron', 'muon']


def load_model(tag):
    Pz = dict(np.load(OUT / f'{tag}_model.npz')); names = PFEAT + NBR + PK
    if tag.startswith('S10'):
        nv = Pz['Wv'].shape[1] // 6
        return dict(kind='joint', names=names, nv=nv, **{k: Pz[k] for k in ('Ws', 'Wv', 'cself', 'bias', 'kns', 'knv', 'mus', 'sds', 'muv', 'sdv')})
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
    npart = ok.sum(1)
    for h in range(16):
        o = O.copy(); o[:, h] = O[:, h].mean(0); drop = a0 - float((logits(o).argmax(1) == ref).mean())
        imp_t = np.abs(Wv[h]) * Pv[:, h].std(0)[:, None]; imp_f = np.zeros(nv)
        for j in range(Wv.shape[1]): imp_f[j % nv] += imp_t[j].sum()
        top = np.argsort(-imp_f)[:6]; curves = {}
        for f in top:
            x = np.linspace(kn[0, f] - (kn[-1, f] - kn[0, f]) * .3, kn[-1, f] + (kn[-1, f] - kn[0, f]) * .3, 40); xs = (x - mu[f]) / sd[f]
            basis = np.stack([xs] + [np.maximum(0, x - kn[k, f]) / sd[f] for k in range(len(kn))], 1); cols = [f] + [nv + k * nv + f for k in range(len(kn))]
            y = basis @ Wv[h][cols]; y -= y.mean(0); curves[vnames[f]] = dict(x=x.round(4).tolist(), y=y.round(4).T.tolist())
        neurons = [[dict(feature=vnames[j % nv], term='linear' if j < nv else f'max(0, x − {kn[(j - nv) // nv, j % nv]:.4g})', coef=float(Wv[h][j, o_]), importance=float(imp_t[j, o_])) for j in np.argsort(-imp_t[:, o_])[:40]] for o_ in range(16)]
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
        heads.append(dict(block=b + 1, head=h % 8 + 1, drop=drop, self_weight=float(A[b, :, h % 8, 0].mean()), eff_particles=float(np.exp(-(al * np.log(al + 1e-12)).sum(1)).mean()),
                          features=[dict(feature=vnames[f], importance=float(imp_f[f])) for f in top], curves=curves, neurons=neurons, selection=sel, rule=rule))
        log(f'  block {b + 1} head {h % 8 + 1}: ablation −{100 * drop:.2f} pt; values use ' + ', '.join(vnames[f] for f in top[:4]))
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
    return dict(tag=tag, agreement=a0, heads=heads, examples=ex, model_js=model_js)


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
    hb = ''.join(f'<button class="tab hb{" on" if i == 0 else ""}" onclick="setHead({i})">block {hd["block"]} · head {hd["head"]} <small>−{100 * hd["drop"]:.1f}</small></button>' for i, hd in enumerate(an['heads']))
    jb = ''.join(f'<span class="btn jb{" on" if i == 0 else ""}" onclick="setJet({i})" style="border-color:{"#2f855a" if e["model"] == e["part"] else "#c05621"}">{CLASSES[e["part"]]} <small>{e["n"]}p</small></span>' for i, e in enumerate(an['examples']))
    panels = ''
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
        nb = ''.join(f'<span class="nbtn{" on" if j == 0 else ""}" onclick="showNeu({i},{j})">{j + 1}</span>' for j in range(16))
        np_ = ''.join(f'<div class="neu{" on" if j == 0 else ""}" id="n{i}_{j}"><pre>{html.escape(neuron_snippet(tag, hd["block"], hd["head"], j, terms))}</pre></div>' for j, terms in enumerate(hd['neurons']))
        panels += f'''<div class="head{" on" if i == 0 else ""}" id="h{i}"><h2>Block {hd["block"]}, head {hd["head"]} <span class="cnt">— removing it: −{100 * hd["drop"]:.2f} pt agreement · class token’s own share {hd["self_weight"]:.2f} · {hd["eff_particles"]:.1f} effective particles · selection: {selmode(hd)}</span></h2>
{rule}<h3>The rule applied to a jet</h3><div class="slot"></div><h3>What the selection does on average</h3><div class="card"><div class="grid">{prof}</div><div class="cnt">{flags}</div></div>
<h3>What it sums over the selected particles: the 16 value neurons</h3><div class="card"><p style="margin:0 0 8px"><b>How to read a neuron:</b> neuron = Σ over <b>every particle i of the jet</b> of α<sub>i</sub> · f(particle i) + c · α<sub>cls</sub> + b. f is the function written below, evaluated on that particle’s inputs; α<sub>i</sub> is this head’s weight for that particle in this jet (the weights of all particles and the class token sum to 1, so it is a weighted average). All 16 neurons of a head share the same weights α and differ in f: 16 properties averaged over the same selection.</p><span class="cnt">inputs by total weight:</span>{fb}<div class="grid">{cv}</div><div style="margin-top:8px"><span class="cnt">neuron:</span> {nb}</div>{np_}</div></div>'''
    js = """const EX=__EX__, CL=__CL__, MJ=__MJ__; let H=0, JJ=0;
function setHead(i){H=i;document.querySelectorAll('.head').forEach((e,k)=>e.classList.toggle('on',k==i));document.querySelectorAll('.hb').forEach((e,k)=>e.classList.toggle('on',k==i));document.querySelectorAll('.head')[i].querySelector('.slot').appendChild(document.getElementById('jetbox'));drawJet();}
function setJet(i){JJ=i;document.querySelectorAll('.jb').forEach((e,k)=>e.classList.toggle('on',k==i));drawJet();}
let NN=0; function setNN(j){NN=j;document.querySelectorAll('.nnb').forEach((e,k)=>e.classList.toggle('on',k==j));drawJet();}
function showNeu(i,j){for(let k=0;k<16;k++){document.getElementById('n'+i+'_'+k).classList.toggle('on',k==j);}document.querySelectorAll('#h'+i+' .nbtn').forEach((e,k)=>e.classList.toggle('on',k==j));}
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
 s+='<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;margin:8px 0"><div>'+jetSVG(e,h,S)+'</div><div class="cnt" style="max-width:360px"><b>'+S.size+' of '+e.n+' particles</b> carry 90 % of this head’s weight on the particles (orange rings; highlighted rows). The class token keeps '+(100*h.self).toFixed(1)+' % for itself. Every particle is included in the sums; the others just count little. The weights α are recomputed for every jet — switch heads to see how differently they look at the same jet. Click a particle to see all its inputs.</div></div>';
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
setHead(0);""".replace('__EX__', json.dumps(an['examples'])).replace('__CL__', json.dumps(CLASSES)).replace('__MJ__', json.dumps(an['model_js']))
    notation_rows = ''.join(f'<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(v)}</td></tr>' for k, v in NOTATION)
    defs_rows = ''.join(f'<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(WORDS.get(k, ""))}</td><td class="cnt">{html.escape(v)}</td></tr>' for k, v in DEFS.items()); m = load_model(tag); py = python_export(tag, m)
    if py: (outdir / f'{tag}_formulas.py').write_text(py)
    shutil.copy(OUT / f'{tag}_model.npz', outdir / f'{tag}_model.npz'); (outdir / 'analysis.json').write_text(json.dumps({k: v for k, v in an.items() if k not in ('examples', 'model_js')}))
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>
<p class="cnt"><a href="../../index.html">← all setups</a> · <a href="../../research/index.html">research log</a></p><h1>{html.escape(title)}</h1>
<p class="cnt">ParT_full · JetClass · ParT’s class attention written per head. A head applies one rule to every jet: a score for each particle → attention weights by softmax (with the class token’s own share) → a sum over the particles of a formula of each particle’s physics (16 value neurons). ParT’s fixed arithmetic after the heads turns the 256 head outputs into the class scores.</p>
<div>{kpi}</div>
<div class="card"><div class="keep"><b>Kept of ParT:</b> {kept}<br><span class="cnt">{DOWN}</span></div><div class="form"><b>Formulas:</b> {formula}</div></div>
<h2>Notation</h2><div class="card"><table>{notation_rows}</table></div>
<h2>What the output is composed of</h2><div class="card"><pre>{html.escape(COMPOSE[joint])}</pre><p class="cnt">Exact formulation with the fitted coefficients: <a href="{tag}_formulas.py">{tag}_formulas.py</a> (+ <a href="{tag}_model.npz">{tag}_model.npz</a>). Each head’s 16 value neurons are written out term by term in its tab.</p></div>
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
