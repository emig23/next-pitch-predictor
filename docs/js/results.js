(function(){
"use strict";
var NPP = window.NPP, D = NPP.D, CLS = NPP.CLS, NAMES = NPP.NAMES, M = NPP.M, $ = NPP.$;
var pr = NPP.pr, pct = NPP.pct, ll = NPP.ll, argmax = NPP.argmax, esc = NPP.esc, fmtDate = NPP.fmtDate, lastName = NPP.lastName;

var RES = [
  {name:'Model (one for all pitchers)', sub:'XGBoost', ll:1.3660, lo:1.3394, hi:1.3922, c:'var(--model)'},
  {name:'Separate model per pitcher', sub:'XGBoost', ll:1.4349, lo:1.3959, hi:1.4758, c:'var(--model)', hollow:true},
  {name:'Baseline C', sub:'pitcher, count, batter hand', ll:1.6256, lo:1.5492, hi:1.7096, c:'var(--basec)'},
  {name:'Baseline B', sub:'pitcher, count', ll:1.6524, lo:1.5877, hi:1.7229, c:'var(--baseb)'},
  {name:'Baseline A', sub:'pitcher only', ll:1.6581, lo:1.6047, hi:1.7160, c:'var(--baseb)', hollow:true}
];
var LO=1.30, HI=1.78;
function xp(v){ return ((v-LO)/(HI-LO)*100).toFixed(2)+'%'; }
$('dp').innerHTML = RES.map(function(r){
  var left=xp(r.lo), w=(((r.hi-r.lo)/(HI-LO))*100).toFixed(2)+'%';
  return '<div class="dp"><div class="lab">'+r.name+'<small>'+r.sub+'</small></div>'+
    '<div class="dtrack"><span class="whisk" style="left:'+left+';width:'+w+'"></span>'+
    '<span class="pt" style="left:'+xp(r.ll)+';background:'+(r.hollow?'var(--surface)':r.c)+';border-color:'+r.c+'"></span></div>'+
    '<div class="num" style="font-weight:600;text-align:right">'+r.ll.toFixed(3)+'</div></div>';
}).join('');
var ticks=''; [1.3,1.4,1.5,1.6,1.7].forEach(function(v){ ticks+='<span class="num" style="left:'+xp(v)+'">'+v.toFixed(1)+'</span>'; });
$('axis').innerHTML=ticks;

var PP = [
  ['Gray, Sonny',2474,1.7609,1.5788,1.5297,1.4741],['Wheeler, Zack',2649,1.7578,1.7256,1.5306,1.4616],
  ['Mikolas, Miles',1826,1.6878,1.6647,1.6170,1.5777],['Kikuchi, Yusei',1202,3.0914,3.5948,4.0941,1.5192],
  ['Gausman, Kevin',3003,1.0884,1.1025,1.1025,0.9739],['Bassitt, Chris',1731,1.7016,1.6590,1.5935,1.5132],
  ['Castillo, Luis',2366,1.3084,1.2896,1.2476,1.1995],['Cease, Dylan',2895,1.7460,1.7182,1.6288,1.4786],
  ['Flaherty, Jack',1658,1.2873,1.2501,1.2416,1.2004],['Webb, Logan',2488,1.6921,1.6527,1.6305,1.4985],
  ['Irvin, Jake',1939,2.2714,2.4323,2.5255,1.5552],['Valdez, Framber',2768,1.3645,1.3074,1.2608,1.1868],
  ['Gallen, Zac',1779,1.6962,1.6789,1.5909,1.3514]
].map(function(r){ return {name:r[0],n:r[1],a:r[2],b:r[3],c:r[4],g:r[5],gain:r[4]-r[5]}; })
 .sort(function(x,y){ return y.gain-x.gain; });
var CAP=0.30, BROKEN=['Kikuchi, Yusei','Irvin, Jake'];
$('gb').innerHTML = PP.map(function(r){
  var cut=r.gain>CAP, w=(Math.min(r.gain,CAP)/CAP*100).toFixed(1)+'%';
  return '<div class="gb"><div class="lab">'+lastName(r.name)+'<small>'+r.n.toLocaleString()+' pitches</small></div>'+
    '<div class="gtrack"><i class="gfill'+(cut?' cut':'')+'" style="width:'+w+'"></i></div>'+
    '<div class="num" style="font-weight:600;text-align:right">+'+r.gain.toFixed(3)+(cut?'*':'')+'</div></div>';
}).join('');
$('callout').innerHTML = '<b>Two pitchers distort the average.</b> For Kikuchi and Irvin, Baseline C scored 4.09 and 2.53, worse than guessing all ten pitches equally (2.30). Their 2026 pitch mix likely differs from their history, which a lookup table cannot adjust to. The model also sees what the pitcher has thrown so far in the game. This explanation is a likely one and has not been tested. *Striped bars run past the scale (0.30).';
function pooled(list, key){ var n=0,s=0; list.forEach(function(r){ n+=r.n; s+=r.n*r[key]; }); return s/n; }
function exclOut(){
  var list = $('excl').checked ? PP.filter(function(r){ return BROKEN.indexOf(r.name)<0; }) : PP;
  var c=pooled(list,'c'), g=pooled(list,'g');
  $('exclout').innerHTML = (list.length)+' pitchers: Baseline C '+c.toFixed(3)+', model '+g.toFixed(3)+', gain <b>+'+(c-g).toFixed(3)+'</b>.'+
    ($('excl').checked ? ' In 2025 the gain was +0.095.' : '');
}
$('excl').addEventListener('change', exclOut); exclOut();

if(D.season){
  var S=D.season, G=S['Global XGBoost'], Cc=S.C, f=function(x){ return (x*100).toFixed(1)+'%'; };
  $('k3n').textContent=f(G.top1);
  $('k3d').textContent='of the time the model\'s top pick was right. The real pitch was in its top 2 '+f(G.top2)+' and its top 3 '+f(G.top3)+'. Baseline C: '+f(Cc.top1)+', '+f(Cc.top2)+', '+f(Cc.top3)+'.';
}
})();
