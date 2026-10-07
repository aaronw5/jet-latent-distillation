"""The page of a formula-SELECTION model (W1 family): ParT's head values and downstream kept, the attention weights
of the 16 heads from formulas — per head a per-particle score formula (what goes into each weight) and a jet-level
formula for the class token's own share. Laid out per head: what the head selects, every score statement as its own
drop-down (meaning of the input, when it fires and on what, what it adds to the score by class, histogram with the
threshold), the class-token share formula, a jet explorer with the formula's scores and weights next to ParT's.

  CTX_HOPS=1 python -m jetdistill.research.alpha_page W1o|W1p OUT_DIR"""
import html, json, shutil, sys, pathlib
import numpy as np
from ..config import RESULTS, CLASSES
from ..pipeline import jets
from ..part.clsfit import PFEAT
from .nbr import NBR, PK, PK2
from .heads import extract, rows_of_split, OUT
from .heads_page import CSS, TYPES, svg_profile
from .heads_defs import DEFS, WORDS, PHRASE, PK_DEF, term_math
from .alpha_fit import REL
from .wloop import setup, Net

RELN = [f'{nm} {k}' for _, nm in REL for k in ('rank in jet', '− jet max', 'z-score in jet')]
NAMES = PFEAT + NBR + PK + PK2 + RELN
INFO = {'W1o': ('Formula selection, ParT’s values — one term per input per head', 'ParT’s head values (what each head sums), its fixed arithmetic after the heads and its last layer.',
                'Which particles each head looks at: per head a score formula of each particle’s physics (own, neighbourhood, pair-kernel context, within-jet relative), softmax over the jet; the class token’s own share from a formula of jet-level quantities. At most one term per input per head. Tuned toward ParT’s probabilities.'),
        'W1p': ('Formula selection, ParT’s values — pruned', None, None)}
INFO['W1p'] = (INFO['W1p'][0], INFO['W1o'][1], INFO['W1o'][2] + ' Pruned to the inputs each head needs, re-tuned.')
HN = lambda h: f'b{h // 8 + 1}h{h % 8 + 1}'


def word(nm):
    if nm in WORDS: return WORDS[nm]
    if ': ' in nm:
        hk, prop = nm.split(': ', 1); base = WORDS.get(prop.replace('ΔR to axis', 'ΔR'), prop)
        kind = 'the pT-weighted neighbourhood' if 'pT-weighted' in hk else 'the neighbourhood of the neighbourhood (2 hops)' if '2-hop' in hk else 'the neighbourhood'
        return f'{base} — averaged over {kind} of particle i as ParT’s pair kernel {hk.split()[0]} sees it' if prop != 'ln ΔR to i' else f'how far (log ΔR) the particles in {kind} of particle i are from it, as kernel {hk.split()[0]} sees it'
    for _, nm0 in REL:
        if nm.startswith(nm0):
            b = WORDS.get(nm0, nm0); k = nm[len(nm0) + 1:]
            return f'{b} — its rank within the jet (0 = largest, as a fraction of n)' if k == 'rank in jet' else f'{b} — minus the largest value in the jet' if k == '− jet max' else f'{b} — in units of the jet’s spread around its mean'
    return nm


def analyze(tag, n_dev=20000, n_ex=6, device='mps', log=print):
    import torch
    from scipy.stats import kendalltau
    S = setup(1000, n_dev, device, log, src=tag); net = Net(S, device); model = S['model']; PA = np.load(OUT / f'{tag}_alpha.npz'); W, CZ, kn, mu, sd = PA['W'], PA['cz'], PA['kn'], PA['mu'], PA['sd']
    nf, K = len(mu), kn.shape[0]; assert nf == len(NAMES), (nf, len(NAMES)); T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)
    Wt, Wzt = T(W), T(CZ.T); a0 = net.agree(Wt, Wzt); log(f'  {tag}: {100 * a0:.2f}% on {n_dev} dev jets')
    J = jets('full', 'dev'); rows = rows_of_split('dev', n_dev); _, A, M, L, _ = extract(model, 'dev', n_dev, device); ref = L.argmax(1); ytrue = np.asarray(J['y'][rows])
    Fd, ok, Zd = S['Fd'], S['okd'], S['Zd']; npart = ok.sum(1); XR = Fd[ok]; YR = np.repeat(ytrue, npart); starts = np.concatenate([[0], np.cumsum(npart)[:-1]]); JR = np.repeat(np.arange(n_dev), npart)
    cols = lambda f: [1 + f] + [1 + nf + k * nf + f for k in range(K)] + [1 + nf + K * nf + k * nf + f for k in range(K)]
    def terms_of(h, f):
        out = []; c = W[:, h]
        if c[1 + f]: out.append((float(c[1 + f]), 'lin', None))
        for k in range(K):
            if c[1 + nf + k * nf + f]: out.append((float(c[1 + nf + k * nf + f]), 'gt', float(kn[k, f])))
            if c[1 + nf + K * nf + k * nf + f]: out.append((float(c[1 + nf + K * nf + k * nf + f]), 'lt', float(kn[k, f])))
        return out
    piece = lambda x, c, k, t, f: c * ((x - mu[f]) / sd[f] if k == 'lin' else np.maximum(0, x - t) / sd[f] if k == 'gt' else np.maximum(0, t - x) / sd[f])
    # the formula's scores and weights of every real particle, per head; ParT's weights
    SC = np.zeros((len(XR), 16), np.float32)
    for h in range(16):
        SC[:, h] = W[0, h]
        for f in range(nf):
            for c, k, t in terms_of(h, f): SC[:, h] += piece(XR[:, f], c, k, t, f)
    mx = np.maximum.reduceat(SC, starts, 0); e = np.exp(SC - mx[JR]); rel = e / np.add.reduceat(e, starts, 0)[JR]
    acls = 1 / (1 + np.exp(Zd @ CZ.T)); AF = ((1 - acls)[JR] * rel).T; AP = np.stack([A[h // 8][:, h % 8, 1:][ok] for h in range(16)]); selfP = np.stack([A[h // 8][:, h % 8, 0] for h in range(16)], 1)
    F0 = Fd[..., :len(PFEAT)]; d0s = np.abs(np.asarray(J['ext'][rows][..., 3], np.float32)) / np.maximum(np.asarray(J['ext'][rows][..., 4], np.float32), 1e-6)
    tp = F0[..., 8:13][ok]; KR = {'charged hadron': tp[:, 0], 'neutral hadron': tp[:, 1], 'photon': tp[:, 2], 'lepton': tp[:, 3] + tp[:, 4]}; ZR = np.exp(F0[..., 2][ok]) > .05
    ncls = np.array([(ytrue == c).sum() for c in range(10)]).clip(1); HIST = {}
    def hist(f):
        if NAMES[f] in HIST: return
        lo_, hi_ = np.quantile(XR[:, f], [.01, .99]); hi_ = hi_ if hi_ > lo_ else lo_ + 1; cnts, edges = np.histogram(XR[:, f], 40, (lo_, hi_)); HIST[NAMES[f]] = dict(edges=[round(float(v), 5) for v in edges], counts=[int(v) for v in cnts])
    # jet-level inputs of the class-token share: names
    keys = [k for k in jets('full', 'fit')['Q']]; Q100 = np.stack([np.asarray(jets('full', 'fit')['Q'][k][:100000], np.float32) for k in keys], 1); okq = np.isfinite(Q100).all(0) & (Q100.std(0) > 0)
    ZN = [k for k, o in zip(keys, okq) if o] + ['ln n particles', '1']; assert len(ZN) == Zd.shape[1], (len(ZN), Zd.shape[1]); zsd = Zd.std(0)
    heads = []; sub = np.sort(np.random.default_rng(0).choice(n_dev, 1500, replace=False))
    for h in range(16):
        b = h // 8; ins = []
        for f in range(nf):
            ts = terms_of(h, f)
            if not ts: continue
            con = sum(piece(XR[:, f], c, k, t, f) for c, k, t in ts); st = []
            for c, k, t in ts:
                fire = np.ones(len(XR), bool) if k == 'lin' else (XR[:, f] > t) if k == 'gt' else (XR[:, f] < t); cf = piece(XR[:, f], c, k, t, f)
                st.append(dict(coef=c, kind=k, thr=t, fires=float(fire.mean()), kinds={k2: float(v2[fire].mean()) if fire.any() else 0.0 for k2, v2 in KR.items()}, hard=float(ZR[fire].mean()) if fire.any() else 0.0,
                               by_class=[float(cf[YR == cc].mean()) if (YR == cc).any() else 0.0 for cc in range(10)], wcorr=float(np.corrcoef(cf, np.log(AF[h] + 1e-9))[0, 1]) if cf.std() > 0 else 0.0))
            hist(f); lo_, hi_ = float(XR[:, f].min()), float(XR[:, f].max()); xs = np.linspace(lo_, hi_, 25); g = sum(piece(xs, c, k, t, f) for c, k, t in ts)
            ins.append(dict(feature=NAMES[f], f=f, importance=float(con.std()), terms=st, direction=1 if (g[-1] - g[0] >= 0 if hi_ > lo_ else ts[0][0] >= 0) else -1))
        ins.sort(key=lambda d: -d['importance'])
        # selection profiles (formula α and ParT's), agreement of the weights
        prof = {}
        for src_, AA in (('formula', AF[h]), ('ParT', AP[h])):
            relw = AA * npart[JR]; sel = {}
            for name, v, bins in (('ln pT/pT_jet', XR[:, 2], np.linspace(-7, -0.5, 9)), ('ΔR', XR[:, 4], np.linspace(0, 0.8, 9)), ('|d0|/σ', d0s[ok], np.array([0, .5, 1, 2, 3, 5, 10, 1e9]))):
                idx = np.clip(np.searchsorted(bins, v) - 1, 0, len(bins) - 2); sel[name] = dict(edges=[float(x) if x < 1e8 else None for x in bins], weight=[float(relw[idx == i].mean()) if (idx == i).any() else None for i in range(len(bins) - 1)])
            for name, msk in (('charged', XR[:, 7] != 0), ('photon', XR[:, 10] > 0), ('lepton', (XR[:, 11] + XR[:, 12]) > 0)): sel[name] = float(relw[msk].mean()) if msk.any() else None
            prof[src_] = sel
        taus = []
        for j in sub:
            sl = slice(starts[j], starts[j] + npart[j])
            if npart[j] >= 3: taus.append(kendalltau(AF[h, sl], AP[h, sl])[0])
        out = []; selp = prof['formula']; w = [v for v in selp['ln pT/pT_jet']['weight'] if v is not None]
        if len(w) > 2 and w[-1] > 1.6 * max(w[0], 1e-9): out.append('the hard particles')
        w = [v for v in selp['ΔR']['weight'] if v is not None]
        if len(w) > 2 and w[0] > 1.4 * max(w[-1], 1e-9): out.append('particles near the axis')
        elif len(w) > 2 and w[-1] > 1.4 * max(w[0], 1e-9): out.append('wide-angle particles')
        for k, nm, th in (('lepton', 'leptons', 1.5), ('charged', 'charged particles', 1.15), ('photon', 'photons', 1.2)):
            if selp.get(k) and selp[k] > th: out.append(nm)
        w = [v for v in selp['|d0|/σ']['weight'] if v is not None]
        if len(w) > 2 and w[-1] > 1.5 * max(w[0], 1e-9): out.append('displaced tracks')
        effF = float(np.exp(-np.add.reduceat(rel[:, h] * np.log(rel[:, h] + 1e-12), starts)).mean()); alP = A[b, :, h % 8, 1:]; effP = float(np.exp(-(alP * np.log(alP + 1e-12)).sum(1)).mean())
        zt = sorted(range(len(ZN)), key=lambda i: -abs(CZ[h, i] * zsd[i]))[:8]
        ph = lambda d: PHRASE.get(d['feature'], ('high ' + d['feature'], 'low ' + d['feature']))[0 if d['direction'] > 0 else 1]
        heads.append(dict(block=b + 1, head=h % 8 + 1, inputs=ins, intercept=float(W[0, h]), selects=', '.join(out[:3]) if out else 'all particles about equally', profiles=prof,
                          tau=float(np.nanmean(taus)), dalpha=float(np.abs(AF[h] - AP[h]).mean() * npart[JR].mean()), eff_formula=effF, eff_part=effP, self_formula=float(acls[:, h].mean()), self_part=float(selfP[:, h].mean()),
                          self_corr=float(np.corrcoef(acls[:, h], selfP[:, h])[0, 1]), cls_terms=[(ZN[i], float(CZ[h, i]), float(CZ[h, i] * zsd[i])) for i in zt],
                          title=(f'weights up {ph(ins[0])}' + (f', {ph(ins[1])}' if len(ins) > 1 else '')) if ins else 'uniform',
                          by_class=[dict(cls=CLASSES[c], effF=float(np.exp(-np.add.reduceat(rel[:, h] * np.log(rel[:, h] + 1e-12), starts))[ytrue == c].mean()), effP=float(np.exp(-(alP * np.log(alP + 1e-12)).sum(1))[ytrue == c].mean()),
                                         selfF=float(acls[ytrue == c, h].mean()), selfP=float(selfP[ytrue == c, h].mean())) for c in range(10)]))
        log(f'  {HN(h)}: {len(ins)} inputs, τ vs ParT {heads[-1]["tau"]:.2f}, selects {heads[-1]["selects"]}')
    # example jets with the formula's scores, weights and per-particle contributions
    with torch.no_grad():
        pred = np.empty(n_dev, int)
        for a in range(0, n_dev, 500):
            i = torch.from_numpy(net.od[a:a + 500]).to(device); pred[net.od[a:a + 500]] = net.logits(net.Fdt[i], net.okdt[i], net.Zdt[i], net.Vdt[i], Wt, Wzt).argmax(1).cpu().numpy()
        Lm = np.zeros((n_dev, 10), np.float32)
        for a in range(0, n_dev, 500):
            i = torch.from_numpy(net.od[a:a + 500]).to(device); Lm[net.od[a:a + 500]] = net.logits(net.Fdt[i], net.okdt[i], net.Zdt[i], net.Vdt[i], Wt, Wzt).cpu().numpy()
    rng = np.random.default_rng(0); ex = []
    for c in range(10):
        cand = np.flatnonzero((ref == c) & (npart >= 8)); pick = list(cand[np.argsort(npart[cand])[:n_ex // 2]]) + list(rng.choice(cand, n_ex - n_ex // 2, replace=False))
        for i in pick:
            ps = np.flatnonzero(ok[i]); sl = slice(starts[i], starts[i] + npart[i])
            parts = [dict(eta=round(float(F0[i, p, 5]), 4), phi=round(float(F0[i, p, 6]), 4), z=float(np.exp(F0[i, p, 2])), dr=float(F0[i, p, 4]), type=TYPES[int(np.argmax(F0[i, p, 8:13]))], q=int(F0[i, p, 7]), d0s=float(d0s[i, p])) for p in ps]
            hs = []
            for h in range(16):
                tops = []
                for p in ps:
                    byf = {}
                    for d in heads[h]['inputs']:
                        f = d['f']; v = sum(piece(Fd[i, p, f], c_, k_, t_, f) for c_, k_, t_ in terms_of(h, f)); byf[d['feature']] = float(v)
                    tj = sorted(byf, key=lambda n_: -abs(byf[n_]))[:5]; tops.append([[n_, round(byf[n_], 2)] for n_ in tj])
                hs.append(dict(s=np.round(SC[sl, h], 3).tolist(), w=np.round(AF[h, sl], 4).tolist(), wp=np.round(AP[h, sl], 4).tolist(), self=round(float(acls[i, h]), 4), selfp=round(float(selfP[i, h]), 4), top=tops))
            sm = np.exp(Lm[i] - Lm[i].max()); sm /= sm.sum(); sp = np.exp(L[i] - L[i].max()); sp /= sp.sum()
            ex.append(dict(part=int(ref[i]), truth=int(ytrue[i]), model=int(pred[i]), p_model=sm.round(3).tolist(), p_part=sp.round(3).tolist(), n=int(npart[i]), particles=parts, heads=hs, X=np.round(Fd[i, ps][:, :38], 4).tolist()))
    return dict(tag=tag, agreement=a0, n_dev=n_dev, heads=heads, examples=ex, hists=HIST, znames=ZN, statements=int(sum(len(d['terms']) for hd in heads for d in hd['inputs'])), pairs=int(sum(len(hd['inputs']) for hd in heads)))


def stmt(c, k, t, feature, f_mu, f_sd):
    v = f'particle["{feature}"]'; s = f'/ {f_sd:.4g}'
    return f'add {c:+.4g} · ({v} − {f_mu:.4g}) {s}' if k == 'lin' else f'if {v} > {t:.4g}:  add {c:+.4g} · ({v} − {t:.4g}) {s}' if k == 'gt' else f'if {v} < {t:.4g}:  add {c:+.4g} · ({t:.4g} − {v}) {s}'


def head_python(hd, mu, sd):
    L = [f'# particle["name"] = that particle’s input called “name” (table of inputs at the bottom of the page)', f'def score_{HN(hd["block"] * 8 - 8 + hd["head"] - 1)}(particle):',
         f'    """head {HN(hd["block"] * 8 - 8 + hd["head"] - 1)}: the score of one particle; its weight = (1 − alpha_cls) * exp(score) / sum over the jet\'s particles of exp(score)"""', f'    s = {hd["intercept"]:.6g}']
    for d in hd['inputs']:
        f = d['f']; v = f'particle["{d["feature"]}"]'
        for st in d['terms']:
            c, k, t = st['coef'], st['kind'], st['thr']
            L.append(f'    s += {c:.6g} * ({v} - {mu[f]:.6g}) / {sd[f]:.6g}' if k == 'lin' else f'    if {v} > {t:.6g}: s += {c:.6g} * ({v} - {t:.6g}) / {sd[f]:.6g}' if k == 'gt' else f'    if {v} < {t:.6g}: s += {c:.6g} * ({t:.6g} - {v}) / {sd[f]:.6g}')
    L.append('    return s'); return '\n'.join(L)


def page(tag, outdir, device='mps', log=print):
    an = analyze(tag, device=device, log=log); title, kept, formula = INFO[tag]; outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True); pct = lambda x: f'{100 * x:.2f}%'
    PA = np.load(OUT / f'{tag}_alpha.npz'); mu, sd = PA['mu'], PA['sd']
    te = next((json.loads(p.read_text()) for p in (OUT / f'{tag}_metrics_full_test.json',) if p.exists()), None)
    if te:
        cl = list(te['rej']); t = '<table><tr><th>model</th><th class="num">same class as ParT</th><th class="num">accuracy</th><th class="num">AUC</th>' + ''.join(f'<th class="num">Rej<sub>{int(te["rej"][c]["eff"] * 1000) / 10:g}%</sub> {c}</th>' for c in cl) + '</tr>'
        for name, mm, ag in (('ParT', te['part'], None), (title, te, te['agreement'])):
            t += f'<tr><td>{html.escape(name)}</td><td class="num">{"—" if ag is None else pct(ag)}</td><td class="num">{mm["accuracy"]:.4f}</td><td class="num">{mm["auc"]:.4f}</td>' + ''.join(f'<td class="num">{mm["rej"][c]["rej"]:.0f}</td>' for c in cl) + '</tr>'
        pc = '<table><tr><th>class</th><th class="num">same class as ParT</th><th class="num">accuracy</th></tr>' + ''.join(f'<tr><td>{CLASSES[p["c"]]}</td><td class="num">{pct(p["agreement"])}</td><td class="num">{pct(p["accuracy"])}</td></tr>' for p in te['per_class']) + '</table>'
        metrics_html = f'<h2>The ParT paper’s metrics (2,000,000 test jets)</h2><div class="card">{t}</table></div><h2>Per class</h2><div class="card">{pc}</div>'
        kpi = f'<span class="kpi">same class as ParT (test)<b>{pct(te["agreement"])}</b></span><span class="kpi">accuracy<b>{te["accuracy"]:.4f}</b><span class="cnt">ParT {te["part"]["accuracy"]:.4f}</span></span><span class="kpi">AUC<b>{te["auc"]:.4f}</b><span class="cnt">ParT {te["part"]["auc"]:.4f}</span></span>'
    else:
        metrics_html = ''; kpi = f'<span class="kpi">same class as ParT<b>{pct(an["agreement"])}</b><span class="cnt">balanced validation sample, {an["n_dev"]:,} jets (test pending)</span></span>'
    kpi += f'<span class="kpi">(head, input) pairs<b>{an["pairs"]}</b><span class="cnt">{an["statements"]} statements</span></span><span class="kpi">rank agreement with ParT’s weights<b>τ {np.mean([hd["tau"] for hd in an["heads"]]):.2f}</b><span class="cnt">mean Kendall τ over the 16 heads</span></span>'
    def bars(vals, lab, fmt, center=None):
        mxv = max(abs(x - (center or 0)) for x in vals) or 1
        return '<table class="bars">' + ''.join(f'<tr><td>{CLASSES[c]}</td><td style="width:150px"><div class="bar" style="width:{130 * abs(v - (center or 0)) / mxv:.0f}px;background:{"#2f855a" if v - (center or 0) >= 0 else "#c05621"}"></div></td><td class="num">{fmt(v)}</td></tr>' for c, v in enumerate(vals)) + f'</table><div class="cnt">{lab}</div>'
    def terms_html(hd):
        out = []
        for d in hd['inputs']:
            feat = d['feature']; f = d['f']
            for st in d['terms']:
                c, k, t = st['coef'], st['kind'], st['thr']; line = stmt(c, k, t, feat, float(mu[f]), float(sd[f]))
                kinds = ', '.join(f'{k2} {100 * v2:.0f} %' for k2, v2 in sorted(st['kinds'].items(), key=lambda q: -q[1]) if v2 > .05)
                byc = sorted(enumerate(st['by_class']), key=lambda q: -abs(q[1]))[:4]; mxc = max(abs(v2) for _, v2 in byc) or 1
                cls = ''.join(f'<tr><td>{CLASSES[ci]}</td><td style="width:120px"><div class="bar" style="width:{100 * abs(v2) / mxc:.0f}px;background:{"#2f855a" if v2 >= 0 else "#c05621"}"></div></td><td class="num">{v2:+.3g}</td></tr>' for ci, v2 in byc)
                when = 'always (a linear term)' if k == 'lin' else f'for {100 * st["fires"]:.0f} % of all particles'
                out.append(f'''<details class="xterm"><summary><code>{html.escape(line)}</code> <span class="cnt">moves the score by ±{d["importance"]:.3g}; corr. with log weight {st["wcorr"]:+.2f}</span></summary>
<div class="grid" style="margin:6px 0 4px 10px"><div><div><b>{html.escape(feat)}</b>: {html.escape(word(feat))} <span class="cnt">({html.escape(DEFS.get(feat, ""))})</span></div>
<div style="margin-top:4px">Adds to the particle’s score (a higher score → a larger share of this head’s weight, through the softmax over the jet). Fires {when}{"" if k == "lin" else f"; those are {kinds}; hard (pT share > 5 %) {100 * st['hard']:.0f} %"}. The amount added grows with the distance from the threshold (in units of the input’s spread {sd[f]:.3g}).</div></div>
<div><div class="cnt">distribution of this input over the particles (threshold in orange)</div><div class="hh" data-f="{html.escape(feat)}" data-thr="{'' if t is None else t}"></div></div>
<div><div class="cnt">what this statement adds to a particle’s score, on average, by true class of the jet</div><table class="bars">{cls}</table></div></div></details>''')
        return ''.join(out) or '<p class="cnt">no inputs: this head weights all particles equally</p>'
    hb = ''.join(f'<button class="tab hb{" on" if i == 0 else ""}" title="{html.escape(hd["title"])}" onclick="setHead({i})">block {hd["block"]} · head {hd["head"]} <small>τ {hd["tau"]:.2f}</small></button>' for i, hd in enumerate(an['heads']))
    jb = ''.join(f'<span class="btn jb{" on" if i == 0 else ""}" onclick="setJet({i})" style="border-color:{"#2f855a" if e["model"] == e["part"] else "#c05621"}">{CLASSES[e["part"]]} <small>{e["n"]}p</small></span>' for i, e in enumerate(an['examples']))
    panels = ''
    for i, hd in enumerate(an['heads']):
        prof = ''.join(f'<div><div class="cnt">weight vs {html.escape(k)} <small>(formula; ×average particle; dashed = 1)</small></div>{svg_profile(v)}<div class="cnt">ParT’s:</div>{svg_profile(hd["profiles"]["ParT"][k])}</div>' for k, v in hd['profiles']['formula'].items() if isinstance(v, dict))
        fb = '<table class="bars">' + ''.join(f'<tr><td>{html.escape(d["feature"])}</td><td style="width:180px"><div class="bar" style="width:{160 * d["importance"] / (hd["inputs"][0]["importance"] or 1):.0f}px"></div></td><td class="num">±{d["importance"]:.2f}</td></tr>' for d in hd['inputs'][:12]) + '</table>'
        czr = ''.join(f'<tr><td>{html.escape(n_)}</td><td class="num">{c:+.3g}</td><td class="num">{e_:+.3g}</td></tr>' for n_, c, e_ in hd['cls_terms'])
        byc = ''.join(f'<tr><td>{x["cls"]}</td><td class="num">{x["effF"]:.1f}</td><td class="num">{x["effP"]:.1f}</td><td class="num">{100 * x["selfF"]:.1f} %</td><td class="num">{100 * x["selfP"]:.1f} %</td></tr>' for x in hd['by_class'])
        hidx = hd['block'] * 8 - 8 + hd['head'] - 1
        panels += f'''<div class="head{" on" if i == 0 else ""}" id="h{i}"><h2>Block {hd["block"]}, head {hd["head"]} <span class="cnt">— {html.escape(hd["title"])} · rank agreement with ParT’s weights τ = {hd["tau"]:.2f} · {len(hd["inputs"])} inputs, {sum(len(d["terms"]) for d in hd["inputs"])} statements</span></h2>
<div class="card"><p style="margin:0 0 6px"><b>What goes into this head’s weight.</b> Every particle i of the jet gets a score s<sub>i</sub> = {hd["intercept"]:+.3g} + the statements below; the head’s weights are α<sub>i</sub> = (1 − α<sub>cls</sub>) · e<sup>s<sub>i</sub></sup> / Σ<sub>j</sub> e<sup>s<sub>j</sub></sup> (softmax over the particles of the jet); the class token keeps α<sub>cls</sub> from the jet-level formula below. ParT then sums its own values of the particles with these weights.</p>
<p style="margin:0 0 6px"><b>Selects</b> {html.escape(hd["selects"])}: {hd["eff_formula"]:.1f} effective particles per jet (ParT’s own weights: {hd["eff_part"]:.1f}); the class token keeps {100 * hd["self_formula"]:.0f} % (ParT {100 * hd["self_part"]:.0f} %, correlation across jets {hd["self_corr"]:.2f}). Mean |Δα| per particle vs ParT, in units of an average particle’s weight: {hd["dalpha"]:.2f}.</p>
<div class="grid"><div><span class="cnt">inputs by how much they move the score:</span>{fb}</div><div><b>Its selection by true class</b><table><tr><th>class</th><th class="num">eff. particles (formula)</th><th class="num">(ParT)</th><th class="num">class-token share (formula)</th><th class="num">(ParT)</th></tr>{byc}</table></div></div></div>
<h3>The score formula, statement by statement</h3><div class="card"><details open><summary><b>If-statements</b> <span class="cnt">(applied to each particle of the jet in turn; score = {hd["intercept"]:+.3g} + the sum of what fires; open a statement for its details)</span></summary>{terms_html(hd)}</details>
<details><summary><b>Formula</b></summary><pre>{html.escape("s(particle) = " + f"{hd['intercept']:+.4g} " + " ".join(term_math(st["coef"], st["kind"], st["thr"], d["feature"]) + f"/{sd[d['f']]:.3g}" for d in hd["inputs"] for st in d["terms"]))}
# linear terms are of (particle["name"] − mean); every term is divided by the input’s spread; α_i = (1 − α_cls) · softmax_i(s)</pre></details>
<details><summary><b>Python code</b> <span class="cnt">(complete, every statement)</span></summary><pre>{html.escape(head_python(hd, mu, sd))}</pre></details></div>
<h3>The class token’s own share, α<sub>cls</sub></h3><div class="card"><p style="margin:0 0 6px">α<sub>cls</sub> = 1 / (1 + e<sup>u</sup>), u = Σ<sub>q</sub> c<sub>q</sub> · z<sub>q</sub>(jet) over jet-level quantities z (quantile-normalized), ln n and 1. The largest terms (coefficient, and coefficient × spread of z):</p><table><tr><th>jet quantity</th><th class="num">c</th><th class="num">c × spread</th></tr>{czr}</table><p class="cnt">A large u → a small class-token share → the particles carry more of the weight.</p></div>
<h3>The rule applied to a jet</h3><div class="slot"></div><h3>What the selection does on average: formula vs ParT</h3><div class="card"><div class="grid">{prof}</div></div></div>'''
    js = """const EX=__EX__, CL=__CL__, HS=__HS__, NAMES=__NAMES__; let H=0, JJ=0;
function histSVG(hs,thr){const w=260,h=70,e=hs.edges,c=hs.counts,mx=Math.max(...c)||1,n=c.length,bw=(w-20)/n,X=v=>10+(w-20)*(v-e[0])/(e[e.length-1]-e[0]+1e-12);
 let g='';c.forEach((v,i)=>{g+='<rect x="'+(10+i*bw).toFixed(1)+'" y="'+(h-14-(h-22)*v/mx).toFixed(1)+'" width="'+(bw-1).toFixed(1)+'" height="'+((h-22)*v/mx).toFixed(1)+'" fill="#c9d4e3"/>';});
 if(thr!==''){const t=parseFloat(thr);g+='<line x1="'+X(t).toFixed(1)+'" x2="'+X(t).toFixed(1)+'" y1="4" y2="'+(h-12)+'" stroke="#c05621" stroke-width="2"/><text x="'+(X(t)+3).toFixed(1)+'" y="12" font-size="10" fill="#c05621">'+t.toPrecision(3)+'</text>';}
 return '<svg viewBox="0 0 '+w+' '+h+'" style="width:100%;max-width:'+w+'px">'+g+'<text x="10" y="'+(h-2)+'" font-size="9" fill="#667">'+e[0].toPrecision(3)+'</text><text x="'+(w-10)+'" y="'+(h-2)+'" font-size="9" text-anchor="end" fill="#667">'+e[e.length-1].toPrecision(3)+'</text></svg>';}
document.addEventListener('toggle',ev=>{const d=ev.target; if(!d.classList||!d.classList.contains('xterm')||!d.open) return; d.querySelectorAll('.hh').forEach(x=>{if(!x.dataset.done){x.innerHTML=histSVG(HS[x.dataset.f],x.dataset.thr);x.dataset.done=1;}});},true);
function setHead(i){H=i;document.querySelectorAll('.head').forEach((e,k)=>e.classList.toggle('on',k==i));document.querySelectorAll('.hb').forEach((e,k)=>e.classList.toggle('on',k==i));document.querySelectorAll('.head')[i].querySelector('.slot').appendChild(document.getElementById('jetbox'));drawJet();}
function setJet(i){JJ=i;document.querySelectorAll('.jb').forEach((e,k)=>e.classList.toggle('on',k==i));drawJet();}
function core(w){const o=w.map((v,k)=>[v,k]).sort((a,b)=>b[0]-a[0]); let acc=0, tot=w.reduce((a,b)=>a+b,0), S=new Set(); for(const [v,k] of o){if(acc>=0.9*tot) break; acc+=v; S.add(k);} return S;}
function jetSVG(e,h,S){const W=360,Hh=300,R=0.8, X=v=>W/2+v/R*(W/2-20), Y=v=>Hh/2-v/R*(Hh/2-20); const mx=Math.max(...h.w);
 let g='<svg viewBox="0 0 '+W+' '+Hh+'" style="width:100%;max-width:'+W+'px;background:#fbfcfe;border:1px solid #e4e7ec;border-radius:8px">';
 [0.2,0.4,0.8].forEach(r=>{g+='<circle cx="'+X(0)+'" cy="'+Y(0)+'" r="'+(r/R*(W/2-20))+'" fill="none" stroke="#e4e7ec" stroke-dasharray="2 3"/><text x="'+(X(r)+2)+'" y="'+(Y(0)-2)+'" font-size="9" fill="#99a">ΔR '+r+'</text>';});
 const ord=e.particles.map((p,k)=>k).sort((a,b)=>h.w[a]-h.w[b]);
 ord.forEach(k=>{const p=e.particles[k], a=h.w[k]/mx, r=3+10*Math.sqrt(p.z); g+='<circle cx="'+X(p.eta)+'" cy="'+Y(p.phi)+'" r="'+r.toFixed(1)+'" fill="rgb('+Math.round(31+(192-31)*(1-a))+','+Math.round(78+(197-78)*(1-a))+','+Math.round(121+(220-121)*(1-a))+')" fill-opacity="'+(0.25+0.75*a).toFixed(2)+'" stroke="'+(S.has(k)?'#c05621':'none')+'" stroke-width="1.6" style="cursor:pointer" onclick="showP('+k+')"><title>particle '+(k+1)+': formula α = '+h.w[k].toFixed(3)+', ParT α = '+h.wp[k].toFixed(3)+'</title></circle>';});
 g+='<text x="8" y="'+(Hh-8)+'" font-size="10" fill="#667">Δη →   (Δφ ↑) · size: pT share · colour: formula α (dark = high) · orange ring: carries 90 % of the weight</text></svg>'; return g;}
function drawJet(){const e=EX[JJ], h=e.heads[H]; let mx=Math.max(h.self,...h.w); const S=core(h.w);
 let s='<div class="cnt">ParT: <b>'+CL[e.part]+'</b> ('+(100*e.p_part[e.part]).toFixed(0)+'%) · this model: <b>'+CL[e.model]+'</b> ('+(100*e.p_model[e.model]).toFixed(0)+'%) · truth '+CL[e.truth]+' · '+e.n+' particles. Head b'+(H<8?1:2)+' h'+(H%8+1)+': score = formula(particle); weight = (1 − α_cls) · softmax over the jet. Click a particle for what set its score.</div>';
 s+='<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;margin:8px 0"><div>'+jetSVG(e,h,S)+'</div><div class="cnt" style="max-width:360px"><b>'+S.size+' of '+e.n+' particles</b> carry 90 % of this head’s weight on the particles (orange rings). The class token keeps '+(100*h.self).toFixed(1)+' % by the formula (ParT: '+(100*h.selfp).toFixed(1)+' %). The last column shows ParT’s own weight for comparison — the formula is judged by whether the same particles come out on top.</div></div>';
 s+='<table><tr><th>#</th><th class="num">pT share</th><th class="num">ΔR</th><th>type</th><th class="num">charge</th><th class="num">|d0|/σ</th><th class="num">score</th><th class="num">weight (formula)</th><th></th><th class="num">ParT’s weight</th></tr>';
 s+='<tr class="cls"><td>class token</td><td></td><td></td><td></td><td></td><td></td><td></td><td class="num">'+h.self.toFixed(3)+'</td><td><span class="wbar" style="width:'+(120*h.self/mx)+'px"></span></td><td class="num">'+h.selfp.toFixed(3)+'</td></tr>';
 e.particles.forEach((p,k)=>{s+='<tr class="p'+(S.has(k)?' core':'')+'" onclick="showP('+k+')"><td>'+(k+1)+'</td><td class="num">'+p.z.toFixed(3)+'</td><td class="num">'+p.dr.toFixed(2)+'</td><td>'+p.type+'</td><td class="num">'+(p.q>0?'+':'')+p.q+'</td><td class="num">'+p.d0s.toFixed(1)+'</td><td class="num">'+h.s[k].toFixed(2)+'</td><td class="num">'+h.w[k].toFixed(3)+'</td><td><span class="wbar" style="width:'+(120*h.w[k]/mx)+'px"></span></td><td class="num">'+h.wp[k].toFixed(3)+'</td></tr>';});
 document.getElementById('jet').innerHTML=s+'</table>'; document.getElementById('contrib').innerHTML='<span class="cnt">click a particle</span>';}
function showP(k){const e=EX[JJ], h=e.heads[H]; document.querySelectorAll('#jet tr.p').forEach((r,i)=>r.classList.toggle('on',i==k));
 let t='<div class="cnt">particle '+(k+1)+': score '+h.s[k].toFixed(2)+' = intercept + the largest contributions by input:</div><pre>'+h.top[k].map(t=>(t[1]>=0?'+':'')+t[1].toFixed(2)+'  '+t[0]).join('\\n')+'\\n  + smaller terms</pre>';
 t+='<div class="cnt">its basic inputs:</div><table>'+NAMES.map((nm,f)=>'<tr><td>'+nm+'</td><td class="num">'+e.X[k][f].toPrecision(4)+'</td></tr>').join('')+'</table>'; document.getElementById('contrib').innerHTML=t;}
setHead(0);""".replace('__EX__', json.dumps(an['examples'])).replace('__CL__', json.dumps(CLASSES)).replace('__HS__', json.dumps(an['hists'])).replace('__NAMES__', json.dumps(NAMES[:38]))
    NOT = [('jet, particles', 'a jet has n particles i = 1…n (up to 128); x_i = the 314 per-particle inputs of particle i: its own physics (18), its neighbourhood (20), the context ParT’s pair kernels give it (264) and 12 within-jet relative values (table at the bottom).'),
           ('score s_hi', 'head h’s score of particle i: s_hi = intercept + Σ terms, each term of one input (x − mean, max(0, x − θ) or max(0, θ − x), divided by the input’s spread). The formula of each head is written out in its tab.'),
           ('class-token share α_h,cls', 'α_h,cls = 1 / (1 + e^{u_h}), u_h a linear formula of jet-level quantities (quantile-normalized), ln n and 1.'),
           ('attention weight α_hi', 'α_hi = (1 − α_h,cls) · e^{s_hi} / Σ_j e^{s_hj}, j over the particles of the jet: the weights of a head over [class token, particles] sum to 1.'),
           ('values, downstream', 'ParT’s own: each head sums its 16 values of the particles (from ParT’s embeddings) with these weights; out-projection, LayerNorms, MLPs, residuals, final LayerNorm and last layer are ParT’s fixed arithmetic.')]
    notation_rows = ''.join(f'<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(v)}</td></tr>' for k, v in NOT)
    used = sorted({d['feature'] for hd in an['heads'] for d in hd['inputs']}, key=NAMES.index)
    defs_rows = ''.join(f'<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(word(k))}</td><td class="cnt">{html.escape(DEFS.get(k, ""))}</td></tr>' for k in used)
    (outdir / 'analysis.json').write_text(json.dumps({k: v for k, v in an.items() if k not in ('examples', 'hists')})); shutil.copy(OUT / f'{tag}_alpha.npz', outdir / f'{tag}_alpha.npz')
    compose = """for every jet:
    X[i, :]            = the 314 per-particle inputs of particle i;  z = jet-level quantities (quantile-normalized), ln n, 1
    for each head h (16):
        s[h, i]        = score_h(X[i, :])                                   # FORMULA (this page): what goes into the weight
        alpha_cls[h]   = 1 / (1 + exp(c_h . z))                             # FORMULA: the class token's own share
        alpha[h, i]    = (1 - alpha_cls[h]) * softmax_i(s[h, :])
        o[h, :]        = sum_i alpha[h, i] * v_h(particle i) + alpha_cls[h] * v_h(class token)   # ParT's values (kept)
    class scores       = ParT_downstream(o)                                 # ParT's fixed arithmetic (kept)"""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>
<p class="cnt"><a href="../../index.html">← all setups</a> · <a href="../../research/index.html">research log</a></p><h1>{html.escape(title)}</h1>
<p class="cnt">ParT_full · JetClass. This page is about the <b>selection</b>: what makes a head of ParT’s class attention give a particle weight. Each head’s attention weights are replaced by a formula of each particle’s physics (a score, softmax over the jet) and a jet-level formula for the class token’s share; what the heads sum (ParT’s values) and everything after is ParT’s own.</p>
<div>{kpi}</div>
<div class="card"><div class="keep"><b>Kept of ParT:</b> {kept}</div><div class="form"><b>Formulas:</b> {formula}</div></div>
<h2>Notation</h2><div class="card"><table>{notation_rows}</table></div>
<h2>What the output is composed of</h2><div class="card"><pre>{html.escape(compose)}</pre><p class="cnt">Coefficients: <a href="{tag}_alpha.npz">{tag}_alpha.npz</a> (W: the 16 score formulas over [1, x, x &gt; θ_k, x &lt; θ_k]; cz: the class-token share formulas; kn, mu, sd). Analysis: <a href="analysis.json">analysis.json</a>.</p></div>
<div id="bar"><span class="cnt">head (and the rank agreement τ of its formula weights with ParT’s):</span><br>{hb}</div>
<div id="jetbox"><div class="card"><div class="cnt">jet (6 per ParT class; green border = this model agrees with ParT on it):</div><div>{jb}</div></div><div class="card" id="jet"></div><div class="card" id="contrib"></div></div>
{panels}
{metrics_html}
<h2>Definitions of the per-particle inputs used</h2><div class="card"><table><tr><th>input</th><th>in words</th><th>definition (math)</th></tr>{defs_rows}</table><p class="cnt">Every input describes <b>one particle i</b> of the jet. “Nearest” means nearest <b>to particle i</b> among the other particles of the same jet, by ΔR. “hk: …” inputs are context through ParT’s pair kernels: {html.escape(PK_DEF)} “2-hop” repeats that averaging once more (the neighbourhood’s neighbourhood); “pT-weighted” weights the neighbours by pT as well. “rank in jet / − jet max / z-score in jet” give an input relative to the other particles of the same jet.</p></div>
</main><script>{js}</script></body></html>"""
    (outdir / 'index.html').write_text(doc); log(f'page: {outdir / "index.html"} {len(doc) // 1024} kB')


if __name__ == '__main__':
    page(sys.argv[1], sys.argv[2])
