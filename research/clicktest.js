const { JSDOM } = require('jsdom'); const fs = require('fs');
const dom = new JSDOM(fs.readFileSync(process.argv[2], 'utf8'), { runScripts: 'dangerously' });
const d = dom.window.document; const errs = []; dom.window.addEventListener('error', e => errs.push(e.message));
const on = sel => [...d.querySelectorAll(sel)].findIndex(e => e.classList.contains('on'));
console.log('head tabs', d.querySelectorAll('.hb').length, '| jets', d.querySelectorAll('.jb').length, '| active head panel', on('.head'));
d.querySelectorAll('.hb')[5].click(); console.log('after clicking head tab 6: active panel', on('.head'), '| jet table rows', d.querySelectorAll('#jet tr').length, '| jetbox inside panel 6:', d.querySelectorAll('.head')[5].contains(d.getElementById('jetbox')));
d.querySelectorAll('.jb')[17].click(); console.log('after clicking jet 18: header', d.querySelector('#jet .cnt').textContent.slice(0, 90));
d.querySelectorAll('#h5 .nbtn:not(.nnb)')[3].click(); console.log('after clicking neuron 4 of head 6: visible neuron', [...d.querySelectorAll('#h5 .neu')].findIndex(e => e.classList.contains('on')));
d.querySelectorAll('#jet tr.p')[0].click(); console.log('particle click (needs score formulas; S5 keeps ParT weights):', d.getElementById('contrib').textContent.slice(0, 80));
console.log('errors:', errs.length ? errs : 'none');
const nn = d.querySelectorAll('.nnb'); if (nn.length) { nn[2].click(); console.log('worked example after neuron 3:', d.querySelector('#jet pre').textContent.replace(/\s+/g, ' ').slice(0, 140)); }
console.log('jet picture circles:', d.querySelectorAll('#jet svg circle').length, '| core rows:', d.querySelectorAll('#jet tr.p.core').length);
d.querySelectorAll('#jet tr.p')[1].click(); console.log('particle 2 inputs shown:', d.getElementById('contrib').querySelectorAll('tr').length, 'rows');
const xt = d.querySelectorAll('details.xterm'); if (xt.length) { xt[0].open = true; xt[0].dispatchEvent(new dom.window.Event('toggle', { bubbles: true })); console.log('statement drop-downs:', xt.length, '| histogram drawn on open:', xt[0].querySelector('.hh svg') !== null); }
