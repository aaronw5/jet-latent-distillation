"""One standalone HTML file of the deck (slides/deck.html), for viewing without the slides app: the slides in deck.json's
order, 1920×1080 each, scaled to the window width; images from slides/images/.
Usage: python slides/build_html.py"""
import json, re
from pathlib import Path

HERE = Path(__file__).parent


def main():
    D = json.loads((HERE / 'deck.json').read_text()); fonts = ''.join(f'<link rel="stylesheet" href="{f["href"]}">' for f in D['faces'].values() if f.get('href'))
    body = ''
    for k, sid in enumerate(D['order'], 1):
        s = (HERE / 'slides' / f'{sid}.html').read_text()
        s = re.sub(r'/_blob/([0-9a-f]{32})', r'images/\1.png', s); s = re.sub(r'<aside>.*?</aside>', '', s, flags=re.S)
        s = s.replace('<section ', '<section class="slide" ', 1)
        body += f'<div class="frame" id="s{k}"><div class="scale">{s}</div></div>'
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{D["title"]}</title>{fonts}
<style>body{{margin:0;background:#d5dae1}}.frame{{position:relative;width:100vw;height:calc(100vw * 1080 / 1920);overflow:hidden;margin:0 0 12px}}
.scale{{position:absolute;top:0;left:0;width:1920px;height:1080px;transform-origin:0 0;transform:scale(calc(100vw / 1920px))}}
.slide{{position:relative;box-sizing:border-box;width:1920px;height:1080px;overflow:hidden}}.slide *{{margin:0;box-sizing:border-box}}
.slide ul,.slide ol{{padding-left:1.2em}}.slide table{{border-collapse:collapse;width:100%}}.slide th,.slide td{{text-align:left;padding:8px 12px;border-bottom:1px solid #d5dae1}}
x-shape{{display:inline-block;flex:none}}</style>
<script>addEventListener('resize',()=>document.querySelectorAll('.scale').forEach(e=>e.style.transform=`scale(${{innerWidth/1920}})`));addEventListener('load',()=>dispatchEvent(new Event('resize')))</script>
</head><body>{body}</body></html>'''
    (HERE / 'deck.html').write_text(html); print(HERE / 'deck.html', len(D['order']), 'slides')


if __name__ == '__main__':
    main()
