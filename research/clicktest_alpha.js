// click test of a formula-selection (α) page
const { JSDOM } = require('jsdom'); const fs = require('fs');
const dom = new JSDOM(fs.readFileSync(process.argv[2], 'utf8'), { runScripts: 'dangerously' });
const d = dom.window.document; const errs = []; dom.window.addEventListener('error', e => errs.push(e.message));
const on = sel => [...d.querySelectorAll(sel)].findIndex(e => e.classList.contains('on'));
console.log('head tabs', d.querySelectorAll('.hb').length, '| jets', d.querySelectorAll('.jb').length, '| statements', d.querySelectorAll('details.xterm').length, '| active head', on('.head'));
d.querySelectorAll('.hb')[5].click(); console.log('after clicking head tab 6: active panel', on('.head'), '| jet rows', d.querySelectorAll('#jet tr.p').length);
d.querySelectorAll('.jb')[17].click(); console.log('after clicking jet 18:', d.querySelector('#jet .cnt').textContent.slice(0, 80));
d.querySelectorAll('#jet tr.p')[0].click(); console.log('particle click:', d.getElementById('contrib').textContent.replace(/\s+/g, ' ').slice(0, 100));
const xt = d.querySelectorAll('details.xterm'); xt[2].open = true; xt[2].dispatchEvent(new dom.window.Event('toggle', { bubbles: true }));
console.log('histogram drawn on open:', xt[2].querySelector('.hh svg') !== null, '| statement:', xt[2].querySelector('summary').textContent.replace(/\s+/g, ' ').slice(0, 90));
console.log('jet picture circles:', d.querySelectorAll('#jet svg circle').length, '| ParT weight column:', /ParT’s weight/.test(d.getElementById('jet').textContent));
console.log('errors:', errs.length ? errs : 'none');
