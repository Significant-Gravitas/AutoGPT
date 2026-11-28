import { FRAME_PENDING_TEXT, SEND_PENDING_ATTRIBUTE } from "./chatFrameHandoff";

// Runs from the server's HTML, before React: Enter or send in the frame's box
// queues the message instead of dropping the key. React takes over once it
// marks the frame hydrated. Kept to plain ES5 because it runs uncompiled.
export const FRAME_HYDRATED_ATTRIBUTE = "data-hydrated";

export const FRAME_SCRIPT = `(function(){
var s=document.currentScript,r=s&&s.parentElement;if(!r)return;
var i=r.querySelector("textarea"),b=r.querySelector("[data-frame-send]");if(!i)return;
function q(){if(r.hasAttribute("${FRAME_HYDRATED_ATTRIBUTE}")||!i.value.trim())return;
r.setAttribute("${SEND_PENDING_ATTRIBUTE}","");var t=r.querySelector('[role="status"]');if(t)t.textContent=${JSON.stringify(FRAME_PENDING_TEXT)}}
i.addEventListener("keydown",function(e){
if(e.key!=="Enter"||e.shiftKey||e.isComposing||e.keyCode===229||r.hasAttribute("${FRAME_HYDRATED_ATTRIBUTE}"))return;
if(window.matchMedia&&matchMedia("(pointer: coarse)").matches)return;
e.preventDefault();q()});
if(b)b.addEventListener("click",q)})();`;
