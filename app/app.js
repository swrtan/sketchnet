const $ = s => document.querySelector(s);
const pad = $('#pad'), ctx = pad.getContext('2d');
let strokes = [], current = null, generation = 0, timer = null;
let busy = false, pending = false, lastSent = 0, activePointer = null;
ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = 16; ctx.strokeStyle = '#fff';
function point(e) {
  const r = pad.getBoundingClientRect();
  return [Math.max(0, Math.min(512, (e.clientX-r.left)*512/r.width)), Math.max(0, Math.min(512, (e.clientY-r.top)*512/r.height))];
}
pad.addEventListener('pointerdown', e => {
  if (current || (e.pointerType === 'mouse' && e.button !== 0)) return;
  if (strokes.length >= 256) return;
  activePointer=e.pointerId; pad.setPointerCapture(e.pointerId);
  current=[point(e)]; strokes.push(current); $('#canvasHint').hidden=true;
  redraw(); queuePredict();
});
pad.addEventListener('pointermove', e => {
  if (!current || e.pointerId!==activePointer) return;
  if (strokes.reduce((n,s)=>n+s.length,0)>=20000) return;
  const p=point(e), q=current[current.length-1];
  if (Math.hypot(p[0]-q[0],p[1]-q[1])<1.5) return;
  current.push(p); ctx.beginPath(); ctx.moveTo(...q); ctx.lineTo(...p); ctx.stroke(); queuePredict();
});
function finish(e) { if(e.pointerId!==activePointer)return; current=null;activePointer=null;queuePredict(true); }
pad.addEventListener('pointerup',finish);pad.addEventListener('pointercancel',finish);
function queuePredict(final=false) {
  if (timer!==null && !final) return;
  clearTimeout(timer);
  timer=setTimeout(()=>{timer=null;predict();},final?0:Math.max(0,180-(performance.now()-lastSent)));
}
function resetReadings() {
  $('#history').dataset.items='';$('#history').textContent='Predictions for this drawing will appear here.';
  $('#preview').removeAttribute('src');$('.activation-grid').replaceChildren();$('#bars').replaceChildren();
  showWaiting('Draw something to begin','Results appear as you sketch');
}
$('#undo').onclick=()=>{
  if(!strokes.length)return;
  current=null;activePointer=null;strokes.pop();generation++;pending=false;redraw();
  if(strokes.length)queuePredict(true);else resetReadings();
};
$('#clear').onclick=()=>{
  clearTimeout(timer);timer=null;strokes=[];current=null;activePointer=null;generation++;pending=false;
  redraw();resetReadings();
};
function redraw() {
  ctx.clearRect(0,0,512,512);
  strokes.forEach(s=>{
    ctx.beginPath();ctx.moveTo(...s[0]);s.slice(1).forEach(p=>ctx.lineTo(...p));ctx.stroke();
    if(s.length===1){ctx.beginPath();ctx.arc(...s[0],8,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();}
  });
  $('#canvasHint').hidden=strokes.length>0;
}
function showWaiting(a,b) {
  $('#results').hidden=true;$('#error').hidden=true;$('#waiting').hidden=false;
  $('#waiting p').replaceChildren(document.createTextNode(a),document.createElement('br'));
  const small=document.createElement('small');small.textContent=b;$('#waiting p').append(small);$('#latency').textContent='—';
}
async function predict() {
  if(!strokes.length)return;
  if(busy){pending=true;return;}
  busy=true;pending=false;const version=generation;
  const snapshot=strokes.map(s=>[s.map(p=>p[0]),s.map(p=>p[1])]);lastSent=performance.now();
  $('#latency').textContent='Running…';
  try {
    const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({strokes:snapshot}),signal:AbortSignal.timeout(10000)});
    const d=await r.json();
    if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:`Prediction unavailable (${r.status})`);
    if(version!==generation)return;
    render(d);
  } catch(e) {
    if(version===generation){$('#waiting').hidden=true;$('#results').hidden=true;$('#error').hidden=false;$('#error').textContent=e.message;$('#latency').textContent='Unavailable';}
  } finally {busy=false;if(pending)queuePredict();}
}
function render(d) {
  $('#waiting').hidden=true;$('#error').hidden=true;
  if(d.empty)return showWaiting('Draw something to begin','Results appear as you sketch');
  $('#results').hidden=false;const top=d.top[0];
  $('#topLabel').textContent=top.label;$('#topProb').textContent=`${(top.probability*100).toFixed(1)}%`;
  $('#latency').textContent=`${d.latency_ms.total.toFixed(1)} ms`;
  $('#latency').title=`Preprocessing ${d.latency_ms.preprocessing.toFixed(2)} ms · model ${d.latency_ms.inference.toFixed(2)} ms · total includes previews, excludes network`;
  $('#bars').innerHTML=d.top.map(({label,probability:p})=>`<div class="bar-row"><b>${label}</b><span class="bar"><i style="width:${p*100}%"></i></span><span>${(p*100).toFixed(1)}%</span></div>`).join('');
  $('#preview').src=d.preview;
  $('.activation-grid').innerHTML=d.activations.map((x,i)=>`<div><img src="${x}" alt="Final convolution channel ${i+1}, independently normalized"></div>`).join('');
  const h=$('#history'),arr=h.dataset.items?JSON.parse(h.dataset.items):[];
  arr.push({label:top.label,prob:top.probability});const bounded=arr.slice(-16);h.dataset.items=JSON.stringify(bounded);
  h.innerHTML=`<div class="history-items">${bounded.map((x,i)=>`<span class="history-item"><small>${i+1}</small><b>${x.label}</b>${(x.prob*100).toFixed(0)}%</span>`).join('')}</div>`;
}
const tabs=[...document.querySelectorAll('.tab')];
function selectTab(b){tabs.forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-selected',String(x===b));x.tabIndex=x===b?0:-1;});document.querySelectorAll('.tabpanel').forEach(x=>x.hidden=x.id!==b.dataset.tab);}
tabs.forEach((b,i)=>{b.onclick=()=>selectTab(b);b.onkeydown=e=>{if(!['ArrowLeft','ArrowRight'].includes(e.key))return;e.preventDefault();const next=tabs[(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length];selectTab(next);next.focus();};});selectTab(tabs[0]);
function lineChart(history) {
  if(!history?.length)return;
  const p=30,W=700,H=240;
  function draw(target,keys,max,colors) {
    const pts=k=>history.map((x,i)=>`${p+i*(W-2*p)/(history.length-1||1)},${H-p-(x[k]??0)/max*(H-2*p)}`).join(' ');
    $(target).innerHTML=`<line x1="30" y1="210" x2="670" y2="210" stroke="#dfe3e6"/>`+keys.map((k,i)=>`<polyline points="${pts(k)}" fill="none" stroke="${colors[i]}" stroke-width="3"/>`).join('')+`<text x="30" y="235" fill="#526075" font-size="12">Epoch 1</text><text x="610" y="235" fill="#526075" font-size="12">${history.length}</text><text x="5" y="25" fill="#526075" font-size="12">${max.toFixed(1)}</text>`;
  }
  draw('#curve',['train_loss','val_loss'],Math.max(...history.map(x=>Math.max(x.train_loss,x.val_loss)),1),['#ec6f55','#17253c']);
  draw('#accuracyCurve',['train_accuracy','val_accuracy'],1,['#ec6f55','#17253c']);
}
async function status() {
  try {
    const r=await fetch('/api/status');if(!r.ok)throw Error('Service unavailable');const d=await r.json();
    const p=$('#statusPill');p.textContent=d.ready?'CPU model ready':'Model unavailable';p.className='pill '+(d.ready?'ready':'fail');
    $('#modelMeta').textContent=d.architecture||'Local checkpoint';
    if(!d.ready)showWaiting('Model is not ready',d.error||'Train a checkpoint, then restart the server.');
    $('#classList').textContent=d.classes.join(' / ');
    $('#valAcc').textContent=d.validation?.accuracy!=null?`${(d.validation.accuracy*100).toFixed(1)}%`:'—';
    $('#testAcc').textContent=d.test?.accuracy!=null?`${(d.test.accuracy*100).toFixed(1)}%`:'—';
    $('#params').textContent=d.parameters?.toLocaleString()??'—';
    $('#checkpoint').textContent=d.checkpoint_bytes?`${(d.checkpoint_bytes/1024).toFixed(0)} KB`:'—';
    $('#cpuTiming').textContent=d.latency?`CPU model median: ${d.latency.inference_median_ms.toFixed(2)} ms · preprocessing: ${d.latency.preprocessing_median_ms.toFixed(2)} ms · total: ${d.latency.total_median_ms.toFixed(2)} ms`:'CPU benchmark not recorded yet.';
    lineChart(d.history);
    const per=d.test?.per_class;
    if(per)$('#perClass').innerHTML='<table><thead><tr><th>Class</th><th>Recall</th><th>Precision</th><th>F1</th></tr></thead><tbody>'+Object.entries(per).map(([name,m])=>`<tr><th>${name}</th><td>${(m.recall*100).toFixed(1)}%</td><td>${(m.precision*100).toFixed(1)}%</td><td>${(m.f1*100).toFixed(1)}%</td></tr>`).join('')+'</tbody></table>';
    if(d.test){document.querySelectorAll('.artifacts img').forEach(img=>{img.src=img.dataset.src;});}
  } catch(e) {$('#statusPill').textContent='Service offline';$('#statusPill').className='pill fail';$('#modelMeta').textContent='Could not reach local API';showWaiting('Service offline','Start the server, then reload this page.');}
}
async function loadGallery(){try{const r=await fetch('/api/gallery');if(!r.ok)throw Error();const d=await r.json();$('#gallery').innerHTML=d.items.map(x=>`<figure><img src="${x.image}" alt="${x.label} Quick Draw reference drawing"><figcaption>${x.label}</figcaption></figure>`).join('')}catch{$('#gallery').innerHTML='<p class="history-empty">Real Quick Draw references are unavailable until the dataset download succeeds. Synthetic fixtures are hidden.</p>'}}
status();loadGallery();
