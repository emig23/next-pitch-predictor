(function(){
"use strict";
var NPP = window.NPP = {};
NPP.D = window.NPP_DATA;
NPP.CLS = NPP.D.classes;
NPP.NAMES = {FF:'Four-seam fastball',SI:'Sinker',FC:'Cutter',SL:'Slider',ST:'Sweeper',CU:'Curveball',KC:'Knuckle curve',CH:'Changeup',FS:'Splitter',SV:'Slurve'};
NPP.M = {g:'Global XGBoost', c:'C', b:'B'};
NPP.$ = function(id){return document.getElementById(id)};

function pr(p, m, k){ return p.p[m][k]/10000; }
function pct(p){ return p>=0.1 ? Math.round(p*100)+'%' : p>=0.005 ? (p*100).toFixed(1)+'%' : '<0.5%'; }
function ll(p){ return -Math.log(Math.max(p,1e-12)); }
function argmax(a){ var b=0; for(var i=1;i<a.length;i++) if(a[i]>a[b]) b=i; return b; }
function esc(s){ return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]}); }
function fmtDate(s){ try{ return new Date(s+'T12:00:00').toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric'}); }catch(e){ return s; } }
function lastName(n){ var p=n.split(', '); return p.length>1 ? p[1]+' '+p[0] : n; }
NPP.pr = pr; NPP.pct = pct; NPP.ll = ll; NPP.argmax = argmax; NPP.esc = esc; NPP.fmtDate = fmtDate; NPP.lastName = lastName;

/* ---------- tabs ---------- */
NPP.hooks = {show:{}, hide:{}};
var TABS = ['replay','results','live'];
NPP.TABS = TABS;
var $ = NPP.$;
NPP.showTab = function(t){
  if(TABS.indexOf(t)<0) t='replay';
  TABS.forEach(function(x){
    $(x).hidden = x!==t;
    $('tab-'+x).setAttribute('aria-selected', x===t ? 'true':'false');
    if(x!==t && NPP.hooks.hide[x]) NPP.hooks.hide[x]();
  });
  if(NPP.hooks.show[t]) NPP.hooks.show[t]();
};
document.querySelectorAll('.tab[data-tab]').forEach(function(b){ b.addEventListener('click',function(){ NPP.showTab(b.dataset.tab); }); });
})();
