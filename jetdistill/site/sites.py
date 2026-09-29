"""The whole site: a landing page comparing the setups, and a selector (setup, particle count) on top of every page.
  site/index.html                 landing page, one section per group of setups (+ the untrained-network control)
  site/<setup>/n<N>/index.html    the page of a setup (site/page.py), with the selector injected
Usage: python -m jetdistill.site.sites [site dir]"""
import json, sys
from pathlib import Path
from ..config import SETUPS, RESULTS
from .page import SITE

GROUPS = [('Main result', ['all']), ('Tuned for agreement', ['all_agree']), ('Mass experiments', ['nophys', 'nomass', 'nomass_strict'])]
NS = (8, 64)
TITLE = 'Distilling the Latent Space of a Real-Time Jet Tagger into Interpretable Physics Equations'
SUMMARY = (('network', 'main', 'tuned on the network'), ('network', 'step4', 'smaller version'), ('neurons', 'step4', 'tuned on the neuron values, smaller'),
           ('labels', 'step4', 'tuned on true labels, smaller'))


def available(site, e, n): return (Path(site) / e / f'n{n}' / 'index.html').exists()


def selector(site, exp, n):
    def link(e, m, text, on):
        if not available(site, e, m): return f'<span style="color:#98a2b3;margin-right:14px">{text} <small>(not run)</small></span>'
        st = 'font-weight:700;color:#1d2433;text-decoration:none;border-bottom:2px solid #1f4e79' if on else 'color:#1f4e79;text-decoration:none'
        return f'<a href="../../{e}/n{m}/index.html" style="{st};margin-right:14px">{text}</a>'
    setups = '<span style="color:#d0d5dd;margin-right:10px">·</span>'.join(''.join(link(e, n, SETUPS[e].label, e == exp) for e in es) for _, es in GROUPS)
    parts = ''.join(link(exp, m, f'{m} particles', m == n) for m in NS)
    return ('<nav style="font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;background:#fff;border-bottom:1px solid #e4e7ec;padding:10px 16px;position:sticky;top:0;z-index:50">'
            f'<a href="../../index.html" style="color:#1f4e79;text-decoration:none;margin-right:18px">← Overview</a><span style="color:#667085;margin-right:8px">Setup:</span>{setups}'
            f'<span style="color:#667085;margin:0 8px 0 10px">|</span>{parts}</nav>')


def summary(e, n):
    base = RESULTS / e / f'n{n}'; reg = json.loads((base / 'formulas.json').read_text()); M = json.loads((base / 'metrics.json').read_text())
    rows = {r['formula']: r for r in M['models'] if r.get('formula')}; out = []
    for fam, role, lab in SUMMARY:
        ts = sorted((t for t, m in reg.items() if m['family'] == fam and m['role'] == role), key=lambda t: -rows[t]['same_as_network'])
        if ts: r = rows[ts[0]]; out.append((lab, r['terms'], r['same_as_network'], r['accuracy']))
    return M['models'][0]['accuracy'], out


def control_html():
    rows = []
    for n in NS:
        f = RESULTS / 'control' / f'n{n}' / 'compare.json'
        if not f.exists(): continue
        C = json.loads(f.read_text()); med = lambda k, key: sorted(r.get(key, 0) for r in C[k])[len(C[k]) // 2]
        rows.append(f'<tr><td>{n} particles</td>' + ''.join(f'<td>{med(k, "r2_test"):.2f}</td><td>{med(k, "auc"):.2f}</td><td>{med(k, "mass_r2"):.2f}</td>' for k in ('trained', 'untrained')) + '</tr>')
    if not rows: return '<p class="cnt">Not run (python -m jetdistill.analysis.control).</p>'
    return ('<p>Step 1 (MARS on the same observables) applied to an untrained network with random weights at the trained layers’ scale and the same number formats. '
            'Median over neurons, test jets: how much of each neuron the formula explains (R²), how well the neuron separates one class from the rest (best AUC), and how much of it the jet mass alone explains.</p>'
            '<table><tr><th></th><th colspan="3">trained network</th><th colspan="3">untrained network</th></tr><tr><th></th>' + '<th>R²</th><th>AUC</th><th>mass</th>' * 2 + '</tr>' + ''.join(rows) + '</table>')


def landing(site):
    def cell(e, n):
        if not available(site, e, n): return '<td class="cnt">not run</td>'
        netacc, fs = summary(e, n)
        lines = ''.join(f'<div><b>{t}</b> if-statements, {lab}: {100 * s:.2f}% same class as the network, accuracy {100 * a:.2f}%</div>' for lab, t, s, a in fs)
        return f'<td><a href="{e}/n{n}/index.html">open the page</a><div class="cnt">network accuracy {100 * netacc:.2f}%</div>{lines}</td>'
    sections = [f'<h2>{name}</h2><table><tr><th style="width:30%">setup</th><th>8 particles</th><th>64 particles</th></tr>'
                + ''.join(f'<tr><td><b>{SETUPS[e].label}</b><div class="cnt">{SETUPS[e].page_desc}</div></td>{"".join(cell(e, n) for n in NS)}</tr>' for e in es) + '</table>' for name, es in GROUPS]
    sections.append(f'<h2>Control: untrained network</h2>{control_html()}')
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title>
<style>:root{{--ink:#1d2433;--mut:#667085;--line:#e4e7ec;--bg:#f7f8fa;--acc:#1f4e79}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:36px 16px 60px}}h1{{margin:0 0 6px;font-size:26px}}h2{{font-size:19px;margin:30px 0 10px}}.cnt{{color:var(--mut);font-size:13px}}a{{color:var(--acc)}}
table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid var(--line);border-radius:10px}}td,th{{padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}th{{background:#f2f4f7;font-weight:600}}
@media(max-width:700px){{td,th{{display:block}}}}</style></head><body><main>
<h1>{TITLE}</h1><div class="cnt">The JEDI-linear jet tagger’s last hidden layer, written as if-statements on jet physics quantities.</div>
<p>A small neural network sorts particle jets into gluon (g), light quark (q), W, Z and top (t). It decides from 16 numbers, its last hidden layer. Each of those numbers is rewritten here as a sum of if-statements on physics quantities of the jet, tuned so that the network’s own last step, fed with them, makes the same decisions as the network, while each formula number stays close to the network’s.</p>
<p>The method was run in several setups that differ in which quantities and thresholds it may use or in what the tuning aims at. Each setup has its own page for 8 and 64 particles per jet.</p>
{"".join(sections)}
<p class="cnt" style="margin-top:24px">All numbers on the whole test file unless stated. “Same class as the network” = share of test jets on which the formula gives the network’s answer.</p>
<p class="cnt">Model: JEDI-linear (github.com/calad0i/JEDI-linear).</p></main></body></html>'''
    (Path(site) / 'index.html').write_text(html)


def main(site=SITE):
    site = Path(site)
    for _, es in GROUPS:
        for e in es:
            for n in NS:
                p = site / e / f'n{n}' / 'index.html'
                if not p.exists(): continue
                h = p.read_text()
                if '<nav ' in h[:h.index('<main>')]: h = h[:h.index('<nav ')] + h[h.index('</nav>') + 6:]
                i = h.index('>', h.index('<body')) + 1; p.write_text(h[:i] + selector(site, e, n) + h[i:])
    landing(site); (site / '.nojekyll').write_text('')


if __name__ == '__main__':
    main(*sys.argv[1:])
