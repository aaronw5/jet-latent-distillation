// click test of a direct-128 page: every control responds, no script errors
const { JSDOM } = require('jsdom'); const fs = require('fs');
const dom = new JSDOM(fs.readFileSync(process.argv[2], 'utf8'), { runScripts: 'dangerously' });
const d = dom.window.document; const errs = []; dom.window.addEventListener('error', e => errs.push(e.message));
console.log('jets', d.querySelectorAll('.jb').length, '| neurons', d.querySelectorAll('details.nd[id^=neu]').length, '| statements', d.querySelectorAll('details.xterm').length, '| heads', d.querySelectorAll('details.nd:not([id])').length);
d.querySelectorAll('.jb')[17].click(); console.log('after clicking jet 18:', d.querySelector('#jet .cnt').textContent.slice(0, 80));
const nopt = d.querySelectorAll('#neusel option'); const pick = +nopt[Math.min(41, nopt.length - 1)].value; dom.window.setNeu(pick); console.log('after choosing output ' + (pick + 1) + ':', new RegExp('neuron ' + (pick + 1)).test(d.querySelector('#jet pre').textContent), '| rows', d.querySelectorAll('#jet tr.p').length);
d.querySelectorAll('#jet tr.p')[1].click(); console.log('particle 2 clicked: tables', d.getElementById('contrib').querySelectorAll('table').length, '| input rows', d.getElementById('contrib').querySelectorAll('table')[1].querySelectorAll('tr').length);
const xt = d.querySelectorAll('details.xterm'); xt[3].open = true; xt[3].dispatchEvent(new dom.window.Event('toggle', { bubbles: true }));
console.log('histogram drawn on open:', xt[3].querySelector('.hh svg') !== null, '| statement summary:', xt[3].querySelector('summary').textContent.replace(/\s+/g, ' ').slice(0, 90));
console.log('jet picture circles:', d.querySelectorAll('#jet svg circle').length, '| class scores line:', /Class scores/.test(d.getElementById('jet').textContent));
const cb = d.querySelectorAll('.cb'); if (cb.length) { cb[3].click(); console.log('class tabs', cb.length, '| after clicking class 4: visible panel', [...d.querySelectorAll('.cpan')].findIndex(e => e.classList.contains('on')), '| neuron links', d.querySelectorAll('.cpan.on a[href^="#neu"]').length); }
console.log('errors:', errs.length ? errs : 'none');
