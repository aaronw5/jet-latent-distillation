// click test of the Try-a-jet page
const { JSDOM } = require('jsdom'); const fs = require('fs');
const dom = new JSDOM(fs.readFileSync(process.argv[2], 'utf8'), { runScripts: 'dangerously' });
const d = dom.window.document; const errs = []; dom.window.addEventListener('error', e => errs.push(e.message));
const W = dom.window; const txt = () => d.getElementById('app').textContent.replace(/\s+/g, ' ');
console.log('jets in the menu', d.querySelectorAll('select option').length, '| head pictures', d.querySelectorAll('#app svg').length);
const sel = d.querySelector('select'); sel.value = '17'; sel.dispatchEvent(new W.Event('change')); console.log('jet 18:', txt().slice(250, 400));
const btn = l => [...d.querySelectorAll('.btn')].find(b => b.textContent === l);
for (const [w, c] of [['ParT’s', 'ParT’s'], ['formulas (W)', 'formulas'], ['ParT’s', 'formulas'], ['formulas (W)', 'ParT’s']]) {
  [...d.querySelectorAll('.btn')].filter(b => b.textContent === w)[0].click(); [...d.querySelectorAll('.btn')].filter(b => b.textContent === c).slice(-1)[0].click();
  console.log(`weights ${w}, content ${c}:`, txt().match(/(ParT itself|S\d+[a-z]*:|W\d+[a-z]*:|C\d+[a-z]*:)[^·]{0,60}/)?.[0], '| neuron drop-downs', d.querySelectorAll('details.nd:not([id])').length);
}
[...d.querySelectorAll('.btn')].filter(b => b.textContent === 'ParT’s')[0].click(); [...d.querySelectorAll('.btn')].filter(b => b.textContent === 'formulas').slice(-1)[0].click();
console.log('neuron drop-downs', d.querySelectorAll('details.nd:not([id])').length, '| statement rows inside', d.querySelectorAll('.tjd tr').length);
[...d.querySelectorAll('div[onclick^="S.oh.add"]')][3].click(); console.log('head 4 opened from its picture:', d.getElementById('hd3').open, '| head drop-downs', d.querySelectorAll('details[id^=hd]').length);
const sel2 = d.querySelector('select'); sel2.value = '5'; sel2.dispatchEvent(new W.Event('change')); console.log('after switching jets head 4 still open:', d.getElementById('hd3').open);
const row = d.querySelector('#hd3 tr[onclick^="selP"]'); if (row) { row.click(); const next = d.querySelector('#hd3 tr[onclick^="selP"][style*="fde8e8"]')?.nextElementSibling; console.log('particle clicked in head 4: drop-down right under its row', !!next && /Why it was chosen/.test(next.textContent), '| red rings', d.querySelectorAll('circle[stroke="#c53030"]').length, '| detail rows elsewhere', d.querySelectorAll('td[colspan="7"]').length); }
console.log('errors:', errs.length ? errs : 'none');
