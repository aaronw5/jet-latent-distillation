"""“Try a jet”: one page to run a jet through ParT and the formula models, mixing the two halves of the class attention.
Weights (which particles each head looks at): ParT's or the selection formulas (W). Content (what is read off the
particles): ParT's or the neuron formulas (S: with ParT's weights; C: with the formula weights). For the chosen jet:
the answer and the class probabilities, why it beat the runner-up, a small picture per head of which particles it
weights, and per neuron every if-statement with whether it fires on this jet and what it adds.

  CTX_HOPS=1 python -m jetdistill.research.jet_page OUT_DIR [S17 W1q C1q]"""
import html, json, sys, pathlib
import numpy as np
from ..config import CLASSES
from .heads import OUT
from .heads_page import CSS
from . import direct_page, alpha_page

HN = lambda h: f'b{h // 8 + 1}h{h % 8 + 1}'


def metric(tag):
    for p, k in ((OUT / f'{tag}_metrics_full_test.json', 'test'),):
        if p.exists(): return dict(value=json.loads(p.read_text())['agreement'], where='2M test jets')
    for name, key in ((f'{tag}_shared.json', 'best'), (f'{tag}_pruned_terms.json', 'final'), (f'{tag}_pruned.json', 'final'), (f'{tag}_one_term.json', 'best'), (f'{tag}_uniform2.json', 'best')):
        p = OUT / name
        if p.exists(): return dict(value=json.loads(p.read_text())[key], where='balanced validation jets')
    return None


def neuron_meta(an):
    return {d['n']: dict(title=d['title'], stmts=d.get('stmts', []), importance=d['importance']) for d in an['neurons'] if d['inputs']}


def build(outdir, s_tag='S17', w_tag='W1q', c_tag='C1q', device='mps', log=print):
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    an_s = direct_page.analyze(s_tag, device=device, log=log); an_w = alpha_page.analyze(w_tag, device=device, log=log)
    an_c = direct_page.analyze(c_tag, device=device, log=log) if (OUT / f'{c_tag}_model.npz').exists() else None
    jets_ = []
    for i, es in enumerate(an_s['examples']):
        ew = an_w['examples'][i]; assert es['n'] == ew['n'] and es['part'] == ew['part'], (i, es['n'], ew['n'])
        j = dict(particles=es['particles'], truth=es['truth'], part=es['part'], p=dict(PP=es['p_part'], PF=es['p_model'], FP=ew['p_model']), cls=dict(PP=es['part'], PF=es['model'], FP=ew['model']),
                 aP=es['A'], selfP=es['self'], aF=[hd['w'] for hd in ew['heads']], selfF=[hd['self'] for hd in ew['heads']], sF=[hd['s'] for hd in ew['heads']], topF=[hd['top'] for hd in ew['heads']],
                 V=dict(PF=es['V']), ST=dict(PF=es['ST']), H=es['H'])
        if an_c:
            ec = an_c['examples'][i]; assert ec['n'] == es['n']
            j['p']['FF'] = ec['p_model']; j['cls']['FF'] = ec['model']; j['V']['FF'] = ec['V']; j['ST']['FF'] = ec['ST']
        jets_.append(j)
    models = dict(PP=dict(name='ParT', desc='ParT itself: its weights and its values.', m=dict(value=1.0, where='by definition')),
                  PF=dict(name=f'{s_tag}: ParT’s weights, formula neurons', desc='the 128 neurons written as per-particle formulas, summed with ParT’s attention weights, then ParT’s last layer', m=metric(s_tag), page=f'../heads_{s_tag}/full/index.html'),
                  FP=dict(name=f'{w_tag}: formula weights, ParT’s values', desc='each head’s weights from a score formula of every particle’s physics (softmax over the jet); what the heads read off the particles is ParT’s', m=metric(w_tag), page=f'../heads_{w_tag}/full/index.html'),
                  FF=dict(name=f'{c_tag}: formula weights, formula neurons', desc='both halves as formulas: the selection formulas pick the particles, the neuron formulas read them; only ParT’s last layer is kept', m=metric(c_tag) if an_c else None, page=f'../heads_{c_tag}/full/index.html', missing=an_c is None))
    meta = dict(PF=neuron_meta(an_s), FF=neuron_meta(an_c) if an_c else {})
    def hdesc(hd):
        tot = sum(d['importance'] for d in hd['inputs']) or 1.0; parts = []
        for d in hd['inputs'][:4]:
            w = alpha_page.word(d['feature']); hi, lo = alpha_page.PHRASE.get(d['feature'], (f'high {d["feature"]}', f'low {d["feature"]}'))
            parts.append(f'{hi if d["direction"] > 0 else lo} ({d["feature"]}: {w}; {100 * d["importance"] / tot:.0f} % of the score’s spread)')
        return dict(selects=hd['selects'], favours=parts, tau=hd['tau'])
    D = dict(jets=jets_, models=models, meta=meta, fc=an_s['fc'], classes=CLASSES, heads=[HN(h) for h in range(16)], hdesc=[hdesc(hd) for hd in an_w['heads']])
    render(outdir, D, log)


def render(outdir, D, log=print):
    """the page from its data (re-render without re-analysing: `python -m ... render OUT_DIR`, data read back from the page)"""
    outdir = pathlib.Path(outdir)
    js = r"""const D=__D__, CL=D.classes; let S={j:0, w:'P', c:'F', oh:new Set(), on:new Set(), p:null, ph:null, why:true};
function selP(k,q){if(S.p===k&&S.ph===q){S.p=null;S.ph=null;}else{S.p=k;S.ph=q;}draw();}
function tg(set,k,el){if(el.open)set.add(k);else set.delete(k);}
function key(){return S.w+S.c}
const COLS=['#1f4e79','#2f855a','#c05621','#6b46c1','#b7791f','#2c7a7b','#c53030','#4a5568','#d53f8c','#3182ce'];
function mini(e,a,self,h,big){const W=big?320:112,Hh=big?280:100,R=0.8,X=v=>W/2+v/R*(W/2-6),Y=v=>Hh/2-v/R*(Hh/2-6),mx=Math.max(...a,1e-9);
 let g='<svg viewBox="0 0 '+W+' '+Hh+'" style="width:'+W+'px;background:#fbfcfe;border:1px solid #e4e7ec;border-radius:6px">';
 const ord=a.map((v,k)=>k).sort((x,y)=>a[x]-a[y]); if(S.p!==null&&S.p<a.length){ord.splice(ord.indexOf(S.p),1);ord.push(S.p);}
 ord.forEach(k=>{const p=e.particles[k],f=a[k]/mx,r=(big?3:1.5)+(big?10:5)*Math.sqrt(p.z),on=S.p===k;g+='<circle cx="'+X(p.eta).toFixed(1)+'" cy="'+Y(p.phi).toFixed(1)+'" r="'+(on?r+(big?3:2):r).toFixed(1)+'" fill="rgb('+Math.round(31+(220-31)*(1-f))+','+Math.round(78+(225-78)*(1-f))+','+Math.round(121+(235-121)*(1-f))+')"'+(on?' stroke="#c53030" stroke-width="'+(big?3:2)+'"':'')+(big?' style="cursor:pointer" onclick="event.stopPropagation();selP('+k+','+h+')"':'')+'><title>particle '+(k+1)+': weight '+a[k].toFixed(3)+(big?' — click to highlight it in every head':'')+'</title></circle>';});
 return g+'</svg>';}
function pdetail(e,k,q){let h='';const pp=e.particles[k],rk=a=>1+a.filter(v=>v>a[k]).length;
  h+='<div style="margin:6px 0;padding:6px 8px;border:1px solid #f5c2c2;background:#fff7f7;border-radius:8px"><b style="color:#c53030">Particle '+(k+1)+'</b> <span class="cnt">— '+pp.type+', pT share '+pp.z.toFixed(3)+', ΔR '+pp.dr.toFixed(2)+', charge '+(pp.q>0?'+':'')+pp.q+', |d0|/σ '+pp.d0s.toFixed(1)+'</span> <span class="btn" onclick="selP('+k+','+q+')">close</span>';
  h+='<div style="overflow-x:auto"><table><tr><th>head</th>'+D.heads.map(x=>'<th class="num">'+x+'</th>').join('')+'</tr><tr><td>ParT’s weight</td>'+e.aP.map(a=>'<td class="num">'+a[k].toFixed(3)+'</td>').join('')+'</tr><tr><td class="cnt">its rank in the jet</td>'+e.aP.map(a=>'<td class="num cnt">'+rk(a)+'</td>').join('')+'</tr><tr><td>formula weight</td>'+e.aF.map(a=>'<td class="num">'+a[k].toFixed(3)+'</td>').join('')+'</tr><tr><td class="cnt">its rank in the jet</td>'+e.aF.map(a=>'<td class="num cnt">'+rk(a)+'</td>').join('')+'</tr></table></div>';
  // why it was chosen: in every head, the selection formula's score for this particle, term by term, against the jet's best
  const top=q=>{const sc=e.sF[q];let b=0;sc.forEach((v,i)=>{if(v>sc[b])b=i;});return b;};
  h+='<details'+(S.why?' open':'')+' ontoggle="S.why=this.open" style="margin-top:6px"><summary><b>Why it was chosen</b> <span class="cnt">— in every head, the selection formula gives each particle a score; the weights are (1 − class-token share) × e<sup>score</sup> / Σ<sub>jet</sub> e<sup>score</sup>, so what matters is its score against the other particles’ scores. The largest terms of its score:</span></summary><div style="overflow-x:auto"><table><tr><th>head</th><th class="num">its score</th><th class="num">rank</th><th class="num">best score in the jet</th><th class="num">its weight (formula / ParT)</th><th>largest terms of its score (input: amount)</th></tr>';
  for(let q2=0;q2<16;q2++){const b=top(q2),sc=e.sF[q2],w=e.aF[q2][k];
   h+='<tr'+(q2===q?' style="background:#fde8e8"':'')+'><td>'+D.heads[q2]+'</td><td class="num"><b>'+sc[k].toFixed(2)+'</b></td><td class="num">'+rk(sc)+' of '+sc.length+'</td><td class="num">'+sc[b].toFixed(2)+(b===k?' (this one)':' (particle '+(b+1)+')')+'</td><td class="num">'+w.toFixed(3)+' / '+e.aP[q2][k].toFixed(3)+'</td><td class="cnt">'+e.topF[q2][k].map(t=>'<span style="color:'+(t[1]>=0?'#2f855a':'#c05621')+'">'+t[0]+(t[2]?' = '+t[2]:'')+': '+(t[1]>=0?'+':'')+t[1].toFixed(2)+'</span>').join('<br>')+'</td></tr>';}
  h+='</table></div><div class="cnt">Green terms raise the particle’s score (more weight), orange terms lower it. Each term is one of the formula’s if-statements on one input (the statements are listed per head on the <a href="'+D.models.FP.page+'">selection-formula page</a>). The score differences matter, not the absolute values: a score 1 above another particle’s means e ≈ 2.7 times its weight.</div></details></div>';return h;}
function bars(p){const mx=Math.max(...p);return CL.map((c,i)=>'<div style="display:flex;align-items:center;gap:6px"><b style="width:44px">'+c+'</b><svg width="180" height="12"><rect x="0" y="1" width="'+(180*p[i]/mx).toFixed(1)+'" height="10" rx="2" fill="'+COLS[i]+'"/></svg><span class="cnt">'+(100*p[i]).toFixed(1)+' %</span></div>').join('');}
function draw(){const e=D.jets[S.j],k=key(),M=D.models[k];
 let h='<div class="card"><b>Try a jet</b> <span class="cnt">— pick a jet, then choose where the weights and the content come from.</span><div style="margin-top:6px">jet: <select onchange="S.j=+this.value;S.p=null;draw()">'+CL.map((c,ci)=>'<optgroup label="ParT says '+c+'">'+D.jets.map((x,i)=>[x,i]).filter(q=>q[0].part===ci).map(([x,i])=>'<option value="'+i+'"'+(i==S.j?' selected':'')+'>'+c+' jet, '+x.particles.length+' particles (truth '+CL[x.truth]+')</option>').join('')+'</optgroup>').join('')+'</select></div>';
 h+='<div style="margin-top:6px">weights (which particles each head looks at): '+[['P','ParT’s'],['F','formulas (W)']].map(([v,l])=>'<span class="btn'+(S.w==v?' on':'')+'" onclick="S.w=\''+v+'\';draw()">'+l+'</span>').join('')+' &nbsp; content (what is read off the particles): '+[['P','ParT’s'],['F','formulas']].map(([v,l])=>'<span class="btn'+(S.c==v?' on':'')+'" onclick="S.c=\''+v+'\';draw()">'+l+'</span>').join('')+'</div></div>';
 if(M.missing||!e.p[k]){document.getElementById('app').innerHTML=h+'<div class="card">This combination ('+M.name+') is still being fitted; it appears here when its loop finishes.</div>';return;}
 const p=e.p[k],win=e.cls[k],ord=p.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]),run=ord[1][1];
 h+='<div class="card" style="display:flex;gap:18px;flex-wrap:wrap"><div style="min-width:280px"><div class="cnt">'+M.name+(M.page?' · <a href="'+M.page+'">its page</a>':'')+'<br>'+M.desc+(M.m?'<br>same class as ParT: <b>'+(100*M.m.value).toFixed(2)+' %</b> ('+M.m.where+')':'')+'</div>';
 h+='<div style="font-size:28px;font-weight:700;color:'+COLS[win]+'">'+CL[win]+' <span style="font-size:14px;color:'+(win==e.part?'#2f855a':'#c53030')+'">'+(win==e.part?'✓ same as ParT':'✗ ParT says '+CL[e.part])+'</span></div><div class="cnt">true class '+CL[e.truth]+' · runner-up '+CL[run]+'</div></div><div>'+bars(p)+'</div></div>';
 // which particles each head weights
 const A=S.w=='P'?e.aP:e.aF, self=S.w=='P'?e.selfP:e.selfF;
 h+='<div class="card"><b>Which particles each head weights</b> <span class="cnt">('+(S.w=='P'?'ParT’s weights':'the selection formulas’ weights')+'; dark = high weight, size = pT share; click a picture to open that head below; click a particle in a head’s table or big picture: it is ringed in every head and its details open under its row)</span>';
 h+='<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px">';
 for(let q=0;q<16;q++)h+='<div style="cursor:pointer;text-align:center" onclick="S.oh.add('+q+');draw();(function(x){if(x&&x.scrollIntoView)x.scrollIntoView();})(document.getElementById(\'hd'+q+'\'))">'+mini(e,A[q],self[q],q,false)+'<div class="cnt">'+D.heads[q]+' · cls '+(100*self[q]).toFixed(0)+' %</div></div>';
 h+='</div>';
 for(let q=0;q<16;q++){const aP=e.aP[q],aF=e.aF[q],o=aP.map((v,i)=>i).sort((x,y)=>A[q][y]-A[q][x]);let acc=0,tot=A[q].reduce((a,b)=>a+b,0),core=0;for(const i of o){if(acc>=0.9*tot)break;acc+=A[q][i];core++;}
  h+='<details class="nd" id="hd'+q+'"'+(S.oh.has(q)?' open':'')+' ontoggle="tg(S.oh,'+q+',this)"><summary><b>Head '+D.heads[q]+'</b> <span class="cnt">— '+core+' of '+e.particles.length+' particles carry 90 % of its weight; class token keeps '+(100*self[q]).toFixed(1)+' % (ParT '+(100*e.selfP[q]).toFixed(1)+' %, formula '+(100*e.selfF[q]).toFixed(1)+' %)</span></summary>';
  if(D.hdesc){const hd=D.hdesc[q];h+='<div class="cnt" style="margin:4px 0"><b>What this head’s selection formula looks for:</b> on average it weights '+hd.selects+'. A particle’s score goes up for '+hd.favours.join('; ')+'. (Its weights rank the particles like ParT’s with Kendall τ = '+hd.tau.toFixed(2)+'.)</div>';}
  h+='<div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:6px"><div>'+mini(e,A[q],self[q],q,true)+'</div><div style="max-height:320px;overflow:auto"><table><tr><th>#</th><th>type</th><th class="num">pT share</th><th class="num">ParT’s weight</th><th class="num">formula weight</th><th class="num">formula score</th><th>why the formula scores it so: its largest terms (input = this particle’s value (the statement’s condition): what it adds to the score)</th></tr>'+o.map(i=>'<tr style="cursor:pointer'+(S.p===i?';background:#fde8e8;font-weight:600':'')+'" onclick="selP('+i+','+q+')"><td>'+(S.p===i&&S.ph===q?'▾ ':'▸ ')+(i+1)+'</td><td>'+e.particles[i].type+'</td><td class="num">'+e.particles[i].z.toFixed(3)+'</td><td class="num">'+aP[i].toFixed(3)+'</td><td class="num">'+aF[i].toFixed(3)+'</td><td class="num">'+e.sF[q][i].toFixed(2)+'</td><td class="cnt">'+e.topF[q][i].slice(0,3).map(t=>'<span style="color:'+(t[1]>=0?'#2f855a':'#c05621')+'">'+t[0]+(t[2]?' = '+t[2]:'')+': '+(t[1]>=0?'+':'')+t[1].toFixed(2)+'</span>').join('<br>')+'</td></tr>'+(S.p===i&&S.ph===q?'<tr><td colspan="7" style="background:#fffafa;border-left:3px solid #c53030">'+pdetail(e,i,q)+'</td></tr>':'')).join('')+'</table></div></div></details>';}
 h+='</div>';
 // content
 if(S.c=='F'){const V=e.V[k],ST=e.ST[k],meta=D.meta[k],FW=D.fc.W;const act=Object.keys(meta).map(Number);
  const dif=act.map(n=>[n,V[n]*(FW[win][n]-FW[run][n])]).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  h+='<div class="card"><b>Why '+CL[win]+' and not '+CL[run]+'</b> <span class="cnt">— the formula neurons, the ones that decide between the two first (the neuron’s value × the difference of its two last-layer weights). Open a neuron for its if-statements on this jet.</span>';
  for(const [n,v] of dif){const m=meta[n],rows=m.stmts.map((s,i)=>[s,ST[n][i]]).sort((a,b)=>Math.abs(b[1][0])-Math.abs(a[1][0]));let tot=0;rows.forEach(r=>tot+=r[1][0]);const nfire=rows.filter(r=>r[1][1]>0).length;
   h+='<details class="nd"'+(S.on.has(n)?' open':'')+' ontoggle="tg(S.on,'+n+',this)"><summary><b>neuron '+(n+1)+'</b> — '+m.title+' <span class="pill" style="background:'+(m.importance=='major'?'#1d2433':m.importance=='moderate'?'#98a2b3':'#e4e7ec')+';color:'+(m.importance=='major'?'#fff':'#1d2433')+'">'+m.importance+'</span> <span class="cnt">value '+V[n].toFixed(2)+' (ParT’s '+e.H[n].toFixed(2)+') · '+(v>=0?'helps '+CL[win]:'helps '+CL[run])+' by '+Math.abs(v).toFixed(2)+' · '+nfire+' of '+rows.length+' statements fire</span></summary>';
   h+='<div class="tjd"><table><tr><th></th><th>if-statement</th><th class="num">fires on</th><th class="num">adds</th></tr>'+rows.map(([s,a])=>'<tr style="'+(a[1]==0?'color:#99a':'')+'"><td>'+(a[1]>0?'✓':'✗')+'</td><td><code>'+s.label.replace(/</g,'&lt;')+'</code></td><td class="num">'+a[1]+' of '+e.particles.length+'</td><td class="num">'+(a[0]>=0?'+':'')+a[0].toFixed(3)+'</td></tr>').join('')+'<tr><td></td><td>class-token terms and bias</td><td></td><td class="num">'+(V[n]-tot>=0?'+':'')+(V[n]-tot).toFixed(3)+'</td></tr><tr><td></td><td><b>neuron '+(n+1)+'</b></td><td></td><td class="num"><b>'+V[n].toFixed(3)+'</b></td></tr></table><div class="cnt">“adds” = Σ over the heads and this jet’s particles of the head’s weight × what the statement gives that particle.</div></div></details>';}
  h+='</div>';}
 else{const FW=D.fc.W,H=e.H;const dif=H.map((v,n)=>[n,v*(FW[win][n]-FW[run][n])]).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,12);
  h+='<div class="card"><b>ParT’s own content</b> <span class="cnt">— the heads sum ParT’s values (from its particle embeddings, not formulas'+(S.w=='F'?', here with the formula weights':'')+'). ParT’s neurons that decide between '+CL[win]+' and '+CL[run]+' (shown for ParT’s own pass):</span><table><tr><th>neuron</th><th class="num">value</th><th class="num">pushes '+CL[win]+' over '+CL[run]+' by</th></tr>'+dif.map(([n,v])=>'<tr><td>'+(n+1)+'</td><td class="num">'+H[n].toFixed(2)+'</td><td class="num">'+v.toFixed(2)+'</td></tr>').join('')+'</table></div>';}
 document.getElementById('app').innerHTML=h;}
draw();""".replace('__D__', json.dumps(D))
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Try a jet</title><style>{CSS}
.tjg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:8px;margin-top:8px}}.tjc{{border:1px solid var(--line);border-radius:8px;padding:6px 8px;cursor:pointer;background:#fff}}.tjc.open{{outline:2px solid var(--acc)}}.tjd{{grid-column:1/-1;border:1px solid var(--line);border-radius:8px;padding:8px;background:#fbfcfe;overflow-x:auto}}</style></head><body><main>
<p class="cnt"><a href="../index.html">← all setups</a> · <a href="../research/index.html">research log</a></p><h1>Try a jet</h1>
<p class="cnt">ParT’s class attention has two halves: <b>which particles</b> each of its 16 heads looks at (the weights) and <b>what</b> it reads off them (the content). Each half can be ParT’s own or a formula. Pick a jet and a combination to see the answer, which particles each head weights, and — with the formula content — which if-statements fire on this jet and what they add.</p>
<div id="app"></div></main><script>{js}</script></body></html>"""
    (outdir / 'index.html').write_text(doc); log(f'page: {outdir / "index.html"} {len(doc) // 1024} kB')


def data_of(page):
    t = pathlib.Path(page).read_text(); i = t.index('const D=') + len('const D='); j = t.index(', CL=D.classes', i); return json.loads(t[i:j])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'render': render(a[1], data_of(pathlib.Path(a[1]) / 'index.html'))
    else: build(a[0], *a[1:])
