"""The research page: the summary table of the experiments (numbers from research/results/*.json) and the full log
(research/LOG.md rendered). Written to OUT_DIR/index.html with the JSON results copied beside it.

  python -m jetdistill.research.page OUT_DIR"""
import html, json, re, shutil, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2] / 'research'; RES = ROOT / 'results'
CSS = """:root{--ink:#1d2433;--mut:#667085;--line:#e4e7ec;--bg:#f7f8fa;--acc:#1f4e79}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:18px 16px 60px}h1{font-size:22px;margin:0 0 6px}h2{font-size:18px;margin:26px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:15px;margin:18px 0 6px}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0;overflow-x:auto}
table{border-collapse:collapse;margin:6px 0}td,th{padding:4px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-weight:600;background:#f2f4f7}.num{text-align:right;font-variant-numeric:tabular-nums}
.cnt{color:var(--mut);font-size:12.5px}a{color:var(--acc)}code{font:12px ui-monospace,Menlo,monospace;background:#f3f5f8;padding:1px 4px;border-radius:4px}
.best{font-weight:700;color:#2f855a}.bad{color:#c05621}ul{margin:4px 0 8px 20px}li{margin:2px 0}.log p{margin:6px 0}"""


def md(text):
    """a small Markdown renderer: headings, bullets (one level), tables, inline code/bold"""
    out, lines, i = [], text.split('\n'), 0
    inl = lambda s: re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', re.sub(r'`(.+?)`', r'<code>\1</code>', html.escape(s)))
    while i < len(lines):
        l = lines[i]
        if l.startswith('#'):
            n = len(l) - len(l.lstrip('#')); out.append(f'<h{min(n + 1, 4)}>{inl(l.lstrip("# "))}</h{min(n + 1, 4)}>'); i += 1
        elif l.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'): rows.append(lines[i]); i += 1
            cells = [[c.strip() for c in r.strip('|').split('|')] for r in rows if not re.match(r'^\|[\s\-|:]+\|$', r)]
            out.append('<table>' + ''.join('<tr>' + ''.join(f'<{"th" if k == 0 else "td"}>{inl(c)}</{"th" if k == 0 else "td"}>' for c in row) + '</tr>' for k, row in enumerate(cells)) + '</table>')
        elif l.startswith('- '):
            items = []
            while i < len(lines) and (lines[i].startswith('- ') or (lines[i].startswith('  ') and items)):
                if lines[i].startswith('- '): items.append(lines[i][2:])
                else: items[-1] += ' ' + lines[i].strip()
                i += 1
            out.append('<ul>' + ''.join(f'<li>{inl(x)}</li>' for x in items) + '</ul>')
        elif l.strip() == '': i += 1
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '- ')): para.append(lines[i].strip()); i += 1
            out.append(f'<p>{inl(" ".join(para))}</p>')
    return '\n'.join(out)


def J(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x): return '—' if x is None else f'{100 * x:.1f}%'


def summary():
    """the headline table: what was replaced by formulas, what was kept of ParT, the agreement"""
    rows = [('jet-level formulas, all_plus (MARS → tune → prune), 1083 terms', 'nothing', 0.8009, 'test 2M; acc 0.7526'),
            ('jet-level formulas, GELU terms, 869 terms', 'nothing', 0.8020, 'test 2M; acc 0.7536'),
            ('class fit: per-particle formulas → ParT’s class blocks', 'class blocks, last layer', 0.7870, 'test 2M; acc 0.7396'),
            ('attention fit: own 2-layer class attention', 'last layer', 0.7539, 'test 2M; acc 0.7117')]
    s5 = J('S5_heads_tuned.json'); s7 = J('S7_weights.json'); s8 = J('S8_heads_uniform2.json'); s10 = J('S10_joint.json'); pr = J('S10_pruned.json'); te = J('S10_model_metrics_full_test.json')
    if s5: rows.append(('per-head formula values (Σ over particles of per-particle physics), tuned', 'ParT’s class-attention weights, downstream', s5['tuned']['best'], f'balanced dev 20k; least squares {pct(s5["all"])}'))
    if s7:
        rows.append(('ParT’s head values; block-1 weights from per-particle score formulas', 'everything else', s7['block1_formula'], 'balanced dev 10k'))
        rows.append(('ParT’s head values; block-1 weights uniform', 'everything else', s7['block1_uniform'], 'balanced dev 10k'))
    if s8: rows.append(('per-head formula values; block 2 a plain average', 'ParT’s block-1 weights, downstream', s8['tuned']['best'], 'balanced dev 20k'))
    for t, name in (('S5', 'per-head formula values, ParT’s weights (S5)'), ('S5rp', 'the same, readable fit and pruned to 380 head–input pairs (S5rp)'),
                    ('S5rpc', 'the same, re-tuned against cancellation between inputs (S5rpc)'), ('S5rpc1', '<b>at most one term per input per neuron</b> (S5rpc1, 6080 terms)')):
        m = J(f'{t}_metrics_full_test.json')
        if m: rows.append((name + ' — 2M test jets', 'ParT’s class-attention weights, downstream', m['agreement'], f'accuracy {m["accuracy"]:.4f}, AUC {m["auc"]:.4f} (ParT {m["part"]["accuracy"]:.4f} / {m["part"]["auc"]:.4f})'))
    s12 = J('S12_heads_S11.json')
    if s12: rows.append(('all formulas, stagewise: α from S11 formulas (ranking × jet-level class-token share), values re-tuned (S12)', 'ParT’s downstream', s12['tuned']['best'], 'balanced dev 20k'))
    if s10: rows.append(('<b>all formulas</b>: block-1 weights and head values from per-particle physics, block 2 a plain average, tuned jointly', 'ParT’s downstream (fixed operations)', s10['best'], f'balanced dev 20k; least squares {pct(s10["least_squares"])}'))
    if pr: rows.append((f'the same, pruned to {len(pr["kept"])} of 126 per-particle features', 'ParT’s downstream', pr['final'], 'balanced dev 20k'))
    if te: rows.append(('the pruned all-formula model on the 2M test jets', 'ParT’s downstream', te['agreement'], f'accuracy {te["accuracy"]:.4f}, AUC {te["auc"]:.4f} (ParT {te["part"]["accuracy"]:.4f} / {te["part"]["auc"]:.4f})'))
    s15, s15o, s15p = J('S15_post.json'), J('S15o_one_term.json'), J('S15p_pruned.json'); t15 = J('S15p_metrics_full_test.json')
    if s15: rows.append(('<b>the 128 class-token neurons directly</b>: per-particle formulas summed with ParT’s weights → ParT’s last layer; no head values, no MLP, no LayerNorms (S15)', 'ParT’s class-attention weights, last layer', s15['best'], f'balanced dev 20k; least squares {pct(s15["least_squares"])}'))
    if s15o: rows.append((f'the same, at most one term per input per neuron and head ({s15o["terms"]} terms) (S15o)', 'ParT’s class-attention weights, last layer', s15o['best'], 'balanced dev 20k'))
    if s15p: rows.append((f'the same, pruned to {s15p["path"][-1]["pairs"]} neuron–input pairs (S15p)', 'ParT’s class-attention weights, last layer', s15p['final'], 'balanced dev 20k'))
    s15q, t15q = J('S15q_pruned_terms.json'), J('S15q_metrics_full_test.json')
    if s15q: rows.append((f'the same, single statements pruned: {s15q["terms"]} statements (S15q)', 'ParT’s class-attention weights, last layer', s15q['final'], 'balanced dev 20k'))
    if t15q: rows.append(('<b>S15q on the 2M test jets</b> — <a href="../heads_S15q/full/index.html">page</a>', 'ParT’s class-attention weights, last layer', t15q['agreement'], f'accuracy {t15q["accuracy"]:.4f}, AUC {t15q["auc"]:.4f} (ParT {t15q["part"]["accuracy"]:.4f} / {t15q["part"]["auc"]:.4f})'))
    tw1 = J('W1p_metrics_full_test.json')
    if tw1: rows.append(('<b>W1p on the 2M test jets</b> — <a href="../heads_W1p/full/index.html">page</a>', 'ParT’s values, downstream', tw1['agreement'], f'accuracy {tw1["accuracy"]:.4f}, AUC {tw1["auc"]:.4f} (ParT {tw1["part"]["accuracy"]:.4f} / {tw1["part"]["auc"]:.4f})'))
    w1q, w2, tw1q, tw2 = J('W1q_pruned.json'), J('W2_uniform2.json'), J('W1q_metrics_full_test.json'), J('W2_metrics_full_test.json')
    if w1q: rows.append((f'the same, pruned further to {w1q["path"][-1]["pairs"]} head–input pairs (W1q)', 'ParT’s values, downstream', w1q['final'], 'balanced dev 20k'))
    if tw1q: rows.append(('<b>W1q on the 2M test jets</b> — <a href="../heads_W1q/full/index.html">page</a>', 'ParT’s values, downstream', tw1q['agreement'], f'accuracy {tw1q["accuracy"]:.4f}, AUC {tw1q["auc"]:.4f}'))
    if w2: rows.append((f'<b>block-1 selection formulas only</b> ({w2["pairs"]} head–input pairs), block 2 a plain average (W2)', 'ParT’s values, downstream', w2['best'], f'balanced dev 20k; before re-tuning {pct(w2["start"])}'))
    if tw2: rows.append(('<b>W2 on the 2M test jets</b> — <a href="../heads_W2/full/index.html">page</a>', 'ParT’s values, downstream', tw2['agreement'], f'accuracy {tw2["accuracy"]:.4f}, AUC {tw2["auc"]:.4f}'))
    a12 = J('A12_probe_split.json')
    if a12: rows.append(('A12 probe: ParT’s class attention with linear keys from a particle’s own physics (ParT’s values)', 'everything but the keys', a12['levels']['own']['keys_only'], f'dev 10k; values predicted instead {pct(a12["levels"]["own"]["values_only"])} — the values need the context, not the selection'))
    a11, p1, s16 = J('A11_direct_alpha.json'), J('P1_positional.json'), J('S16_post.json')
    if s16: rows.append(('the 128 neurons directly without the class-token share (S16)', 'ParT’s class-attention weights (particles only), last layer', s16['best'], 'balanced dev 20k'))
    if a11: rows.append(('the 128 neurons directly with the formula selection, everything tuned (A11)', 'last layer', a11['best'], f'balanced dev 20k; neurons refit {pct(a11["least_squares"])}'))
    if p1: rows.append(('no attention: the 12 hardest + 6 most displaced particles in fixed order → 128 neurons (P1)', 'last layer', p1['best'], f'balanced dev 20k; least squares {pct(p1["least_squares"])}'))
    if t15: rows.append(('<b>S15p on the 2M test jets</b> — <a href="../heads_S15p/full/index.html">page</a>', 'ParT’s class-attention weights, last layer', t15['agreement'], f'accuracy {t15["accuracy"]:.4f}, AUC {t15["auc"]:.4f} (ParT {t15["part"]["accuracy"]:.4f} / {t15["part"]["auc"]:.4f})'))
    a2h, a10 = J('A2h_alpha_e2e.json'), J('A10_fulltune.json'); w1o, w1p = J('W1o_one_term.json'), J('W1p_pruned.json')
    if a2h: rows.append(('ParT’s head values; both blocks’ weights from per-particle score formulas tuned end to end (A2h)', 'ParT’s values, downstream', a2h.get('part_values_both', a2h.get('best')), 'balanced dev 20k'))
    if w1o: rows.append((f'the same, at most one term per input per head ({w1o["terms"]} terms), re-tuned (W1o)', 'ParT’s values, downstream', w1o['best'], 'balanced dev 20k'))
    if w1p: rows.append((f'the same, pruned to {w1p["path"][-1]["pairs"]} head–input pairs (W1p)', 'ParT’s values, downstream', w1p['final'], 'balanced dev 20k'))
    if a10: rows.append(('all formulas, per head: coefficients and thresholds tuned together (A10)', 'ParT’s downstream', a10['best'], f'balanced dev 20k; start {pct(a10["start"])}'))
    rows = [r for r in rows if r[2] is not None]; best = max(r[2] for r in rows)
    t = '<table><tr><th>model</th><th>kept of ParT</th><th class="num">same class as ParT</th><th>where measured</th></tr>'
    for name, kept, ag, where in rows: t += f'<tr><td>{name}</td><td>{kept}</td><td class="num{" best" if ag == best else ""}">{pct(ag)}</td><td class="cnt">{where}</td></tr>'
    return t + '</table>'


def page(outdir):
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True); (outdir / 'results').mkdir(exist_ok=True)
    for f in RES.glob('*.json'): shutil.copy(f, outdir / 'results' / f.name)
    log_html = md((ROOT / 'LOG.md').read_text())
    files = ''.join(f'<a href="results/{f.name}">{f.name}</a> ' for f in sorted(RES.glob('*.json')))
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reproducing ParT: research log</title><style>{CSS}</style></head><body><main>
<p class="cnt"><a href="../index.html">← all setups</a></p><h1>Reproducing ParT with formulas: the research loop</h1>
<p class="cnt">ParT_full on JetClass. Agreement = the formula model’s class is ParT’s. Every experiment, its setup, numbers and conclusion, in the order run;
code on branch <code>part-features</code>, <code>jetdistill/research/</code>.</p>
<h2>Where it stands</h2><div class="card">{summary()}
<p class="cnt">Method in the JEDI-linear spirit: formulas of physics quantities fitted to the network, tuned toward its probabilities, pruned. ParT’s neurons come
from class attention, so the formulas are per head: sums over particles of per-particle physics functions, weighted by the head’s attention; the operations
after the heads (out-projection, LayerNorms, MLP, residuals, last layer) are ParT’s own fixed arithmetic.</p></div>
<h2>Result files</h2><div class="card cnt">{files}</div>
<h2>The log</h2><div class="card log">{log_html}</div>
</main></body></html>"""
    (outdir / 'index.html').write_text(doc); print('page:', outdir / 'index.html', f'{len(doc) // 1024} kB')


if __name__ == '__main__':
    page(sys.argv[1])
