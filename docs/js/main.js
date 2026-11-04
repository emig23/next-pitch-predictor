(function(){
"use strict";
var h=(location.hash||'').replace('#','');
NPP.showTab(NPP.TABS.indexOf(h)>=0 ? h : 'replay');
})();
