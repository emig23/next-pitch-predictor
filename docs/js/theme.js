(function(){
"use strict";
var $ = window.NPP.$;
function isDark(){
  var t=document.documentElement.getAttribute('data-theme');
  if(t) return t==='dark';
  return !!(window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches);
}
function paintBtn(){ $('themeBtn').textContent = isDark() ? 'Light' : 'Dark'; }
try{ var saved=localStorage.getItem('npp-theme'); if(saved==='light'||saved==='dark') document.documentElement.setAttribute('data-theme',saved); }catch(e){}
$('themeBtn').addEventListener('click',function(){
  var next = isDark() ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme',next);
  try{ localStorage.setItem('npp-theme',next); }catch(e){}
  paintBtn();
});
paintBtn();
})();
