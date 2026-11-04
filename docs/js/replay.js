(function(){
"use strict";
var NPP = window.NPP, D = NPP.D, CLS = NPP.CLS, NAMES = NPP.NAMES, M = NPP.M, $ = NPP.$;
var pr = NPP.pr, pct = NPP.pct, ll = NPP.ll, argmax = NPP.argmax, esc = NPP.esc, fmtDate = NPP.fmtDate, lastName = NPP.lastName;

var pitchers = [];
D.games.forEach(function(g){ if(pitchers.indexOf(g.name)<0) pitchers.push(g.name); });
var st = {name:pitchers[0], gi:0, i:0, rev:false};
var timer = null;

function gamesOf(name){ var r=[]; D.games.forEach(function(g,i){ if(g.name===name) r.push(i); }); return r; }
function cur(){ return D.games[st.gi]; }

var sel = $('pitcher');
sel.innerHTML = pitchers.map(function(n){ return '<option value="'+esc(n)+'">'+esc(lastName(n))+'</option>'; }).join('');
sel.addEventListener('change', function(){ setPitcher(sel.value); });

function setPitcher(n){ st.name=n; sel.value=n; setGame(gamesOf(n)[0]); }
function setGame(gi){ stopAuto(); st.gi=gi; st.i=0; st.rev=false; renderGameButtons(); renderAll(); }

function teamInfo(g){
  var f=g.pitches[0], home = f.top===1;
  var mine = home ? g.home_team : g.away_team, opp = home ? g.away_team : g.home_team;
  return {mine:mine, opp:opp, home:home};
}
function renderGameButtons(){
  var ids = gamesOf(st.name);
  $('gbtns').innerHTML = ids.map(function(gi){
    var g=D.games[gi], t=teamInfo(g);
    return '<button class="gbtn" data-gi="'+gi+'" aria-pressed="'+(gi===st.gi)+'">'+fmtDate(g.game_date)+'<small>'+(t.home?'vs ':'at ')+esc(t.opp)+' · '+g.pitches.length+' pitches</small></button>';
  }).join('');
  document.querySelectorAll('.gbtn').forEach(function(b){ b.addEventListener('click',function(){ setGame(+b.dataset.gi); }); });
}

function seenCount(){ return st.rev ? st.i+1 : st.i; }

function gameStats(g, upto){
  var out={}, k;
  ['g','c','b'].forEach(function(m){ out[m]={loss:0,hit:0}; });
  for(var j=0;j<upto;j++){
    var p=g.pitches[j], a=CLS.indexOf(p.actual);
    ['g','c','b'].forEach(function(m){
      var arr=p.p[M[m]];
      out[m].loss += ll(arr[a]/10000);
      if(argmax(arr)===a) out[m].hit++;
    });
  }
  out.n=upto; return out;
}

function renderSituation(){
  var g=cur(), p=g.pitches[st.i];
  var inning = (p.top?'Top ':'Bottom ')+p.inning;
  var prev=[]; for(var j=st.i-1;j>=0 && g.pitches[j].ab===p.ab;j--) prev.unshift(g.pitches[j].actual);
  var T = teamInfo(g);
  var score = (p.fs!=null) ? T.mine+' '+p.fs+' – '+p.bs+' '+T.opp :
              (p.diff>0 ? T.mine+' leading by '+p.diff : p.diff<0 ? T.mine+' trailing by '+(-p.diff) : 'Tied');
  var balls='', strikes='', outs='', i;
  for(i=0;i<3;i++) balls+='<i class="dot ball'+(i<p.balls?' on':'')+'"></i>';
  for(i=0;i<2;i++) strikes+='<i class="dot strike'+(i<p.strikes?' on':'')+'"></i>';
  for(i=0;i<3;i++) outs+='<i class="dot'+(i<p.outs?' on':'')+'"></i>';
  var b1=p.on[0], b2=p.on[1], b3=p.on[2];
  var diamond='<svg class="diamond" viewBox="0 0 96 80" role="img" aria-label="Runners on: '+((b1||b2||b3)?[b1?'first':'',b2?'second':'',b3?'third':''].filter(Boolean).join(', '):'none')+'">'+
    '<path class="base'+(b2?' on':'')+'" d="M48 6 L60 18 L48 30 L36 18 Z"/>'+
    '<path class="base'+(b3?' on':'')+'" d="M20 34 L32 46 L20 58 L8 46 Z"/>'+
    '<path class="base'+(b1?' on':'')+'" d="M76 34 L88 46 L76 58 L64 46 Z"/>'+
    '<path class="base" d="M48 50 L58 60 L48 70 L38 60 Z" style="fill:var(--surface-2)"/></svg>';
  var hand = p.stand==='L' ? 'Left' : 'Right';
  $('sit').innerHTML =
    '<div class="sit-top"><div><div class="eyebrow">Situation</div><div class="inning">'+inning+'</div><div class="score num" aria-label="Score">'+score+'</div></div>'+diamond+'</div>'+
    '<div class="stats">'+
      '<div class="stat"><div class="lbl">Balls</div><div class="dots" aria-label="'+p.balls+' balls">'+balls+'</div></div>'+
      '<div class="stat"><div class="lbl">Strikes</div><div class="dots" aria-label="'+p.strikes+' strikes">'+strikes+'</div></div>'+
      '<div class="stat"><div class="lbl">Outs</div><div class="dots" aria-label="'+p.outs+' outs">'+outs+'</div></div>'+
      '<div class="stat"><div class="lbl">Batter</div><div class="v">'+hand+'-handed</div></div>'+
    '</div>'+
    '<div><div class="lbl eyebrow">Earlier pitches this at-bat</div><div class="prev">'+
      (prev.length ? prev.map(function(c){ return '<span class="chip" title="'+NAMES[c]+'">'+c+'</span>'; }).join('') : '<span class="note">None. First pitch of the at-bat.</span>')+
    '</div></div>'+
    '';
}

function renderRows(){
  var g=cur(), p=g.pitches[st.i], a=CLS.indexOf(p.actual), top=argmax(p.p[M.g]);
  $('rows').innerHTML = CLS.map(function(c,k){
    var pg=pr(p,M.g,k);
    var cls='row'+(st.rev&&k===a?' thrown':'')+(st.rev&&k===top?' pick':'');
    return '<li class="'+cls+'" aria-label="'+NAMES[c]+': '+pct(pg)+' chance'+(st.rev&&k===a?'. Thrown.':'')+'">'+
      '<span><span class="code">'+c+'</span><span class="nm">'+NAMES[c]+'</span><span class="flag">'+(k===a?'✓ thrown':'')+'</span></span>'+
      '<span class="track"><i class="bar" style="width:'+(pg*100).toFixed(2)+'%"></i></span>'+
      '<span class="val num">'+pct(pg)+'</span></li>';
  }).join('');
}

function renderResult(){
  var g=cur(), p=g.pitches[st.i], a=CLS.indexOf(p.actual), top=argmax(p.p[M.g]);
  var el=$('result');
  if(!st.rev){ el.innerHTML=''; return; }
  var pg=pr(p,M.g,a);
  var rk=rankOf(p.p[M.g],a);
  var tail = rk===1 ? 'It was the forecast\'s top pick.' : 'It was the forecast\'s '+ordinal(rk)+' choice. The top pick was a '+NAMES[CLS[top]].toLowerCase()+' ('+pct(pr(p,M.g,top))+').';
  el.innerHTML = '<h3>He threw a '+NAMES[p.actual].toLowerCase()+'.</h3>'+
    '<div class="three" style="grid-template-columns:auto 1fr;align-items:center"><div class="big num">'+pct(pg)+'</div><div>chance it was given before the pitch. '+tail+'</div></div>';
}

function rankOf(arr,a){ var r=1; for(var i=0;i<arr.length;i++) if(i!==a && (arr[i]>arr[a] || (arr[i]===arr[a] && i<a))) r++; return r; }
function ordinal(r){ return r===2?'2nd':r===3?'3rd':r+'th'; }
function renderSoFar(){
  var g=cur(), n=seenCount(), h1=0,h2=0,h3=0;
  for(var j=0;j<n;j++){ var p=g.pitches[j], a=CLS.indexOf(p.actual), r=rankOf(p.p[M.g],a); if(r<=1)h1++; if(r<=2)h2++; if(r<=3)h3++; }
  function cell(h,label){ return '<div><div class="big num">'+(n?Math.round(h/n*100)+'%':'–')+'</div><div class="d">'+label+(n?' · '+h+' of '+n:'')+'</div></div>'; }
  $('sofar').innerHTML = '<div class="eyebrow">This game so far · '+n+' of '+g.pitches.length+' pitches revealed</div><div class="so3">'+
    cell(h1,'Top pick was right')+cell(h2,'Thrown pitch was in the top 2')+cell(h3,'Thrown pitch was in the top 3')+'</div>'+
    '<p class="note" style="margin-top:10px">Two games per pitcher, picked by a fixed rule and not for how well they went.</p>';
}

function renderStrip(){
  var g=cur(), seen=seenCount(), html='';
  g.pitches.forEach(function(p,j){
    var cls='cell', a=CLS.indexOf(p.actual);
    if(j<seen) cls += argmax(p.p[M.g])===a ? ' hit':' miss';
    if(j===st.i) cls+=' cur';
    if(j>0 && g.pitches[j-1].ab!==p.ab) cls+=' ab';
    html+='<button class="'+cls+'" data-j="'+j+'" aria-label="Pitch '+(j+1)+(j<seen?(argmax(p.p[M.g])===a?', top pick right':', top pick wrong'):'')+'"></button>';
  });
  $('strip').innerHTML=html;
  document.querySelectorAll('.cell').forEach(function(b){ b.addEventListener('click',function(){ stopAuto(); st.i=+b.dataset.j; st.rev=false; renderAll(); }); });
}

function renderHead(){
  var g=cur(), t=teamInfo(g);
  $('gname').textContent = lastName(g.name);
  $('gmeta').textContent = 'Pitching for '+t.mine+', '+(t.home?'home vs ':'away at ')+t.opp+' · '+fmtDate(g.game_date)+' · throws '+(g.throws==='L'?'left':'right');
  $('pcount').textContent = 'Pitch '+(st.i+1)+' of '+g.pitches.length;
  $('fctitle').textContent = st.rev ? 'What he threw' : 'What will he throw?';
  $('fchelp').textContent = st.rev ? 'The outlined row is the pitch he threw. The underlined code was the forecast\'s top pick.' : 'Each bar is the chance he throws that pitch next, made before the pitch. Press the button, or Space, to see what he threw.';
}

function renderControls(){
  var g=cur(), end = st.i===g.pitches.length-1 && st.rev;
  var nb=$('next');
  nb.textContent = end ? 'Replay this game' : st.rev ? 'Next pitch' : 'Reveal pitch';
  $('back').disabled = (st.i===0 && !st.rev);
  $('auto').textContent = timer ? 'Pause' : 'Auto-play';
}

function renderAll(){ renderHead(); renderSituation(); renderRows(); renderResult(); renderSoFar(); renderStrip(); renderControls(); }

function next(){
  var g=cur();
  if(!st.rev){ st.rev=true; }
  else if(st.i<g.pitches.length-1){ st.i++; st.rev=false; }
  else { stopAuto(); st.i=0; st.rev=false; }
  renderAll();
}
function back(){
  if(st.rev){ st.rev=false; } else if(st.i>0){ st.i--; st.rev=true; }
  renderAll();
}
function stopAuto(){ if(timer){ clearInterval(timer); timer=null; } if($('auto')) $('auto').textContent='Auto-play'; }
function startTimer(){
  timer=setInterval(function(){
    var g=cur();
    if(st.i===g.pitches.length-1 && st.rev){ stopAuto(); renderControls(); return; }
    next();
  }, +$('speed').value);
}
function toggleAuto(){
  if(timer){ stopAuto(); return; }
  var g=cur();
  if(st.i===g.pitches.length-1 && st.rev){ st.i=0; st.rev=false; renderAll(); }
  startTimer();
  renderControls();
}
$('speed').addEventListener('change', function(){ if(timer){ clearInterval(timer); startTimer(); } });
$('next').addEventListener('click', function(){ stopAuto(); next(); });
$('back').addEventListener('click', function(){ stopAuto(); back(); });
$('auto').addEventListener('click', toggleAuto);
document.addEventListener('keydown', function(e){
  if($('replay').hidden) return;
  var tag=(e.target.tagName||'').toLowerCase();
  if(tag==='select'||tag==='input') return;
  if(e.key==='ArrowRight'){ e.preventDefault(); stopAuto(); next(); }
  else if(e.key==='ArrowLeft'){ e.preventDefault(); stopAuto(); back(); }
  else if(e.key===' ' && tag!=='button'){ e.preventDefault(); stopAuto(); next(); }
});
NPP.hooks.hide.replay = stopAuto;
setPitcher(pitchers[0]);
})();
