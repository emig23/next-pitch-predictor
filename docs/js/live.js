(function(){
"use strict";
var NPP = window.NPP, D = NPP.D, CLS = NPP.CLS, NAMES = NPP.NAMES, M = NPP.M, $ = NPP.$;
var pr = NPP.pr, pct = NPP.pct, ll = NPP.ll, argmax = NPP.argmax, esc = NPP.esc, fmtDate = NPP.fmtDate, lastName = NPP.lastName;
NPP.hooks.show.live = function(){ liveInit(); };
var API = 'https://next-pitch-predictor.onrender.com';
var live = {started:false, ready:false, pitchers:[], prev:[], seq:0, timer:null};

function liveStatus(kind, html, retry){
  var el = $('liveStatus');
  if(!kind){ el.innerHTML=''; return; }
  el.innerHTML = '<div class="banner'+(kind==='err'?' err':'')+'">'+html+(retry?'<br><button class="btn sec" type="button" id="liveRetry">Try again</button>':'')+'</div>';
  if(retry) $('liveRetry').addEventListener('click', function(){ live.started=false; liveInit(); });
}
function apiFetch(path, opts, ms){
  var ctl = new AbortController(), t = setTimeout(function(){ ctl.abort(); }, ms);
  opts = opts || {}; opts.signal = ctl.signal;
  return fetch(API+path, opts).then(function(r){ clearTimeout(t); return r; }, function(e){ clearTimeout(t); throw e; });
}
function liveInit(){
  if(live.started) return;
  live.started = true;
  var slow = setTimeout(function(){
    liveStatus('info','<span class="spin"></span>Waking up the server. The free host puts it to sleep when nobody is using it, so the first request can take up to a minute.');
  }, 2500);
  liveStatus('info','<span class="spin"></span>Connecting to the prediction API…');
  apiFetch('/pitchers', {}, 90000).then(function(r){
    if(!r.ok) throw new Error('HTTP '+r.status);
    return r.json();
  }).then(function(list){
    clearTimeout(slow);
    live.pitchers = list.filter(function(p){ return p.in_model; }).sort(function(a,b){ return a.name<b.name?-1:1; });
    if(!live.pitchers.length) throw new Error('no pitchers');
    $('lPitcher').innerHTML = live.pitchers.map(function(p,i){ return '<option value="'+i+'">'+esc(lastName(p.name))+' ('+p.throws+'HP)</option>'; }).join('');
    liveStatus(null);
    $('liveBody').hidden = false;
    live.ready = true;
    livePredict();
  }).catch(function(){
    clearTimeout(slow);
    live.started = false;
    liveStatus('err','<strong>The prediction API isn’t responding right now.</strong> It may be down or still starting up. The Game replay and Results tabs don’t depend on it and work as normal.', true);
  });
}
function seg(id){ var v=null; $(id).querySelectorAll('button').forEach(function(b){ if(b.getAttribute('aria-pressed')==='true') v=b.dataset.v; }); return v; }
function segSet(id, v){ $(id).querySelectorAll('button').forEach(function(b){ b.setAttribute('aria-pressed', b.dataset.v===v?'true':'false'); }); }
function deriveCount(){
  var b=0, s=0;
  live.prev.forEach(function(p){
    if(p.result==='ball') b=Math.min(3,b+1);
    else if(p.result==='foul') s=Math.min(2,s+1);
    else s=Math.min(2,s+1);
  });
  return {balls:b, strikes:s};
}
function renderPrev(){
  $('lPrev').innerHTML = live.prev.length ? live.prev.map(function(p){ return '<span class="chip" title="'+NAMES[p.pitch_type]+', '+p.result.replace('_',' ')+'">'+p.pitch_type+' · '+p.result.replace('_',' ')+'</span>'; }).join('') : '<span class="note">None. First pitch of the at-bat.</span>';
  var has = live.prev.length>0;
  $('lBalls').disabled = has; $('lStrikes').disabled = has;
  $('lUndo').disabled = !has;
  $('lAdd').disabled = live.prev.length>=12;
  if(has){ var c=deriveCount(); $('lBalls').value=c.balls; $('lStrikes').value=c.strikes; }
  $('lPrevNote').textContent = has ? 'The count follows the pitches you add. Undo the last one to set it by hand. (A foul with two strikes keeps the count.)' : '';
}
function buildRequest(){
  var pit = live.pitchers[+$('lPitcher').value];
  var prev = live.prev.map(function(p,i){ return {at_bat_number:1, pitch_number:i+1, batter_id:1, pitch_type:p.pitch_type, result:p.result}; });
  return {
    pitcher_id: pit.pitcher_id, pitcher_throws: pit.throws, batter_id: 1, batter_stands: seg('lStand'),
    at_bat_number: 1, pitch_number: prev.length+1, inning: +$('lInning').value, inning_half: seg('lHalf'),
    balls: +$('lBalls').value, strikes: +$('lStrikes').value, outs: +$('lOuts').value,
    runner_on_1b: $('lR1').checked, runner_on_2b: $('lR2').checked, runner_on_3b: $('lR3').checked,
    pitching_team_score: Math.max(0,Math.min(30,Math.round(+$('lMine').value||0))),
    batting_team_score: Math.max(0,Math.min(30,Math.round(+$('lOpp').value||0))),
    previous_pitches: prev
  };
}
function livePredict(){
  if(!live.ready) return;
  clearTimeout(live.timer);
  live.timer = setTimeout(function(){
    var body = buildRequest(), my = ++live.seq;
    $('lReq').textContent = 'POST '+API+'/predict\n'+JSON.stringify(body,null,2);
    $('lState').innerHTML = '<span class="spin"></span>Asking the model…';
    apiFetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)}, 60000).then(function(r){
      return r.json().then(function(j){ if(!r.ok) throw new Error(r.status===422?'invalid':'http'); return j; });
    }).then(function(res){
      if(my!==live.seq) return;
      $('lState').textContent = '';
      var top = res.probabilities[0];
      $('lTop').innerHTML = '<span class="big">'+esc(top.pitch_type)+'</span><span><strong>'+esc(top.name)+'</strong> is the model’s top pick at '+pct(top.probability)+'.</span>';
      $('lRows').innerHTML = res.probabilities.map(function(p,i){
        return '<li class="row'+(i===0?' pick':'')+'" aria-label="'+esc(p.name)+': '+pct(p.probability)+' chance">'+
          '<span><span class="code">'+esc(p.pitch_type)+'</span><span class="nm">'+esc(p.name)+'</span></span>'+
          '<span class="track"><i class="bar" style="width:'+(p.probability*100).toFixed(2)+'%"></i></span>'+
          '<span class="val num">'+pct(p.probability)+'</span></li>';
      }).join('');
    }).catch(function(e){
      if(my!==live.seq) return;
      $('lState').textContent = '';
      if(e.message==='invalid') $('lState').textContent = 'That situation was rejected as invalid.';
      else { live.ready=false; live.started=false; $('liveBody').hidden=true; liveStatus('err','<strong>Lost contact with the prediction API.</strong> The Game replay and Results tabs still work.', true); }
    });
  }, 250);
}
(function(){
  var inn=''; for(var i=1;i<=9;i++) inn+='<option>'+i+'</option>'; $('lInning').innerHTML=inn;
  $('lPt').innerHTML = CLS.map(function(c){ return '<option value="'+c+'">'+c+' · '+NAMES[c]+'</option>'; }).join('');
  ['lPitcher','lBalls','lStrikes','lOuts','lInning','lMine','lOpp'].forEach(function(id){ $(id).addEventListener('change', livePredict); });
  ['lMine','lOpp'].forEach(function(id){ $(id).addEventListener('input', livePredict); });
  ['lR1','lR2','lR3'].forEach(function(id){ $(id).addEventListener('change', livePredict); });
  ['lStand','lHalf'].forEach(function(id){ $(id).querySelectorAll('button').forEach(function(b){ b.addEventListener('click', function(){ segSet(id, b.dataset.v); livePredict(); }); }); });
  $('lAdd').addEventListener('click', function(){ live.prev.push({pitch_type:$('lPt').value, result:$('lRes').value}); renderPrev(); livePredict(); });
  $('lUndo').addEventListener('click', function(){ live.prev.pop(); renderPrev(); livePredict(); });
  renderPrev();
})();
})();
