// Draws every tab of a built page in Node with a minimal DOM stub and prints any script error:  node tests/page_test.js site/<setup>/n<N>
const fs=require('fs'),path=require('path');const dir=process.argv[2];
const html=fs.readFileSync(path.join(dir,'index.html'),'utf8');const js=html.slice(html.lastIndexOf('<script>')+8,html.lastIndexOf('</script>'));
const els={};const mk=id=>({id,_h:'',set innerHTML(v){this._h=v},get innerHTML(){return this._h},hidden:false,style:{},dataset:{},textContent:'',appendChild(){},remove(){},scrollIntoView(){},querySelectorAll(){return []},addEventListener(){},classList:{add(){},remove(){},toggle(){}}});
global.document={getElementById:id=>els[id]||(els[id]=mk(id)),querySelector:s=>mk(s),querySelectorAll:()=>[],createElement:t=>mk(t),addEventListener(){},body:mk('body'),title:''};
global.window={self:1,top:1,scrollTo(){},addEventListener(){}};global.location={pathname:'/x/index.html',hash:''};global.localStorage={getItem(){return null},setItem(){}};
const pend=[];global.fetch=f=>{const p=Promise.resolve({json:async()=>JSON.parse(fs.readFileSync(path.join(dir,f),'utf8')),text:async()=>fs.readFileSync(path.join(dir,f),'utf8')});pend.push(p);return p};
global.requestAnimationFrame=f=>f();global.setTimeout=(f)=>{try{f()}catch(e){console.log('ERR timeout',e.message)}};
process.on('unhandledRejection',e=>{console.log('ERR promise',e.stack.split('\n').slice(0,3).join(' | '))});
try{(0,eval)(js+';global.__t=TABS;global.__show=k=>{TAB=k;show()}')}catch(e){console.log('ERR load',e.stack.split('\n').slice(0,3).join(' | '))}
(async()=>{for(let r=0;r<2;r++)for(const [k] of global.__t||[]){try{global.__show(k);await new Promise(r=>setImmediate(r));await Promise.all(pend)}catch(e){console.log('ERR tab',k,e.stack.split('\n').slice(0,3).join(' | '))}
 const el=els[k==='made'?'made':k];if(r==1)console.log(k,'chars',(el&&el._h||'').length)}})();
