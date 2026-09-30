// node grab.js <url> <steps.json> <outdir>: runs each step {name, js, el, wait} in headless Chrome, saves an element screenshot per step
const {spawn}=require('child_process'),fs=require('fs');const [url,stepsF,out]=process.argv.slice(2);const steps=JSON.parse(fs.readFileSync(stepsF,'utf8'));
const chrome=spawn(process.env.CHROME||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--remote-debugging-port=9334','--window-size=1300,1400','--hide-scrollbars','--user-data-dir='+out+'/.chrome','about:blank'],{stdio:'ignore'});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{let tgt;for(let i=0;i<40;i++){try{tgt=(await (await fetch('http://127.0.0.1:9334/json')).json()).find(t=>t.type==='page');if(tgt)break}catch(e){}await sleep(250)}
 const ws=new WebSocket(tgt.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let id=0;const pend={};ws.onmessage=m=>{const d=JSON.parse(m.data);if(d.id&&pend[d.id]){pend[d.id](d);delete pend[d.id]}};
 const send=(method,params={})=>new Promise(r=>{const i=++id;pend[i]=r;ws.send(JSON.stringify({id:i,method,params}))});
 const ev=async e=>{const r=await send('Runtime.evaluate',{expression:e,awaitPromise:true,returnByValue:true});if(r.result.exceptionDetails)throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0,400));return r.result.result.value};
 await send('Page.enable');await send('Emulation.setDeviceMetricsOverride',{width:1300,height:1400,deviceScaleFactor:2,mobile:false});await send('Page.navigate',{url});
 for(let i=0;i<100&&await ev('document.readyState')!=='complete';i++)await sleep(100);await sleep(400);
 for(const s of steps){try{if(s.js)await ev(`(async()=>{${s.js}})()`);await sleep(s.wait||400);
  const r=await ev(`(()=>{let e=(${s.el});if(!e)return null;const A=Array.isArray(e)?e:[e];if(!A.length)return null;A[0].scrollIntoView({block:'start'});window.scrollBy(0,-20);let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;for(const a of A){const b=a.getBoundingClientRect();if(!b.width&&!b.height)continue;x0=Math.min(x0,b.left);y0=Math.min(y0,b.top);x1=Math.max(x1,b.right);y1=Math.max(y1,b.bottom)}return {x:x0+scrollX,y:y0+scrollY,w:x1-x0,h:y1-y0}})()`);
  if(!r){console.log(s.name,'NOT FOUND');continue}await sleep(200);const p=s.pad??8;
  const sh=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:Math.max(0,r.x-p),y:Math.max(0,r.y-p),width:r.w+2*p,height:Math.min(r.h+2*p,s.maxh||4000),scale:1}});
  fs.writeFileSync(`${out}/${s.name}.png`,Buffer.from(sh.result.data,'base64'));console.log(s.name,Math.round(r.w),'x',Math.round(r.h))}catch(e){console.log(s.name,'ERR',e.message.slice(0,300))}}
 ws.close();chrome.kill();process.exit(0)})().catch(e=>{console.log('FAIL',e);chrome.kill();process.exit(1)});
