function Ajax(e){this.method=e.method||"",this.url=e.url||"",this.callback=e.callback||"",this.data=e.data||""}var trans={chooseLan:"cmcm",optionVal:{cmcm:{},google:{},baidu:{},youdao:{}},queryVal:"",init:function(){var e=this;e.bindEvent(),e.searchHas()},bindEvent:function(){function e(){for(var e=0,t=h.length;e<t;e++)h[e].className=""}var t=this,a=document.querySelector(".copy"),o=document.querySelector(".after-lan"),n="200",l=!1;a.onclick=function(){t.oCopy(o)};var r=document.querySelector(".before-lan"),i=document.querySelector(".clear");r.oninput=function(e){if(e.target.value?i.style.display="block":(i.style.display="none",o.value=""),n<e.target.scrollHeight){var a=e.target.scrollHeight+"px";e.target.style.height=a,o.style.height=a}t.queryVal=e.target.value},i.onclick=function(){r.value="",i.style.display="none",p.value="",t.optionVal.google.data="",t.optionVal.cmcm.data="",t.optionVal.youdao.data="",t.optionVal.baidu.data="",o.style.height=n+"px",r.style.height=n+"px",l=!1};var c=document.querySelector(".submit"),u=document.querySelector(".before-opt"),d=document.querySelector(".before-lan"),s=document.querySelector(".after-opt"),p=document.querySelector(".after-lan");c.onclick=function(e){if(!d.value)return!1;e.preventDefault();var a=document.querySelector(".on");text=a.innerHTML,a.click(),t.optionVal[text].canSubmit=!0};var p=document.querySelector(".after-lan"),y=document.querySelector(".de-trans"),h=y.getElementsByTagName("li");y.addEventListener("click",function(a){if(a.target&&"LI"==a.target.nodeName){t.tab=!0;var o=a.target.getAttribute("data-value");e(),a.target.className="on";var n={from:u.value,query:t.queryVal,to:s.value,engine:""};if(!d.value)return!1;if(t.optionVal[o].type==t.queryVal&&""!=t.optionVal[o].data)return p.value=t.optionVal[o].data,!1;p.value="",t.optionVal[o].data="",p.setAttribute("placeholder","玩命加载中...");var l="http://10.60.242.202:8090/trans";switch(o){case"google":t.chooseLan="google",n.engine=t.chooseLan,n.query=t.queryVal;var r=new Ajax({method:"post",url:l,callback:function(e){var e=e.trans;t.optionVal.google.data=e,t.optionVal.google.type=t.queryVal,p.value=e,p.setAttribute("placeholder","")},data:n});r.send();break;case"youdao":t.chooseLan="youdao",n.engine=t.chooseLan;var r=new Ajax({method:"post",url:l,callback:function(e){var e=e.trans;t.optionVal.youdao.data=e,t.optionVal.youdao.type=t.queryVal,p.value=e,p.setAttribute("placeholder","")},data:n});r.send();break;case"cmcm":t.chooseLan="cmcm",n.engine=t.chooseLan;var r=new Ajax({method:"post",url:l,callback:function(e){var e=e.trans;t.optionVal.cmcm.data=e,t.optionVal.cmcm.type=t.queryVal,p.value=e,p.setAttribute("placeholder","")},data:n});r.send();break;case"baidu":t.chooseLan="baidu",n.engine=t.chooseLan;var r=new Ajax({method:"post",url:l,callback:function(e){var e=e.trans;t.optionVal.baidu.data=e,t.optionVal.baidu.type=t.queryVal,p.value=e,p.setAttribute("placeholder","")},data:n});r.send()}}});var f=window.location.href,v=f.indexOf("?")>0?f.split("?")[1].split("=")[1]:"";if("debug"==v)for(var m=document.querySelectorAll(".debug"),g=0,b=m.length;g<b;g++)m[g].style.display="block"},oCopy:function(e){e.value&&(e.select(),document.execCommand("Copy"))},saveTextAsFile:function(e){var t=e,a=new Blob([t],{type:"text/plain"}),o="english.txt",n=document.createElement("a");n.download=o,n.innerHTML="Download File",null!=window.webkitURL?n.href=window.webkitURL.createObjectURL(a):(n.href=window.URL.createObjectURL(a),n.onclick=destroyClickedElement,n.style.display="none",document.body.appendChild(n)),n.click()},searchHas:function(){}};Ajax.prototype.send=function(e,t,a,o){var e=e||this.method,o=o||this.data,t=t||this.url,a=a||this.callback,n=new XMLHttpRequest;if(n.onreadystatechange=function(){if(4===n.readyState)if(200===n.status){var e=JSON.parse(n.responseText);a(e)}else console.log(n)},"get"===e){if("object"==typeof o){var l="?";for(var r in o)l+=r+"="+o[r],l+="&";l=l.slice(0,-1)}n.open(e,t+l,!0),n.send(null)}else{if("post"!==e)return console.log("不识别的方法:"+e),!1;n.open(e,t,!0),n.setRequestHeader("Content-Type","application/x-www-form-urlencoded");var i="";for(var c in o)i&&(i+="&"),i+=c+"="+o[c];n.send(i)}};var siblings=function(e){for(var t=[],a=e;e=e.previousSibling;)if(1===e.nodeType){t.push(e);break}for(e=a;e=e.nextSibling;)if(1===e.nodeType){t.push(e);break}return t};window.onload=function(){trans.init()};