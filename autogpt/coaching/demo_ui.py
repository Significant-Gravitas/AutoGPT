"""HTML template for the ABN Co-Navigator demo page.

Served at GET /demo — embeddable in Wix via an HTML component or iframe.
Template variables injected at request time:
  {api_base}      — Railway service URL (e.g. https://abn.up.railway.app)
  {demo_key}      — COACHING_DEMO_KEY value
  {coach_name}    — COACHING_COACH_NAME value
  {calendly_url}  — COACHING_COACH_CALENDLY_URL value
"""

# NOTE: JavaScript braces are doubled ({{ }}) to escape Python's str.format().
DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ABN Consulting – AI Co-Navigator</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#f0f4f8;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
/* ── Header ── */
.hdr{{background:#1a2b4a;color:#fff;padding:10px 18px;display:flex;
  align-items:center;gap:10px;flex-shrink:0}}
.hdr-logo{{font-size:17px;font-weight:700;letter-spacing:-.3px}}
.hdr-sub{{font-size:11px;opacity:.7}}
.hdr-badge{{margin-left:auto;background:#2ecc71;font-size:11px;
  padding:3px 8px;border-radius:10px;font-weight:600}}
/* ── Screens ── */
#landing,#chat-screen,#summary-screen{{flex:1;display:flex;
  flex-direction:column;overflow:hidden}}
#chat-screen,#summary-screen{{display:none}}
/* ── Landing ── */
.lnd{{flex:1;display:flex;flex-direction:column;align-items:center;
  justify-content:center;padding:32px 20px;gap:16px}}
.lnd-icon{{width:60px;height:60px;background:#1a2b4a;border-radius:50%;
  display:flex;align-items:center;justify-content:center;font-size:26px}}
.lnd-title{{font-size:22px;font-weight:700;color:#1a2b4a;text-align:center}}
.lnd-desc{{font-size:14px;color:#6b7280;text-align:center;
  max-width:360px;line-height:1.6}}
.name-row{{display:flex;gap:8px;width:100%;max-width:360px}}
input[type=text]{{flex:1;padding:11px 14px;border:1.5px solid #d1d5db;
  border-radius:10px;font-size:14px;outline:none;transition:border-color .2s}}
input[type=text]:focus{{border-color:#1a2b4a}}
.btn{{background:#1a2b4a;color:#fff;border:none;padding:11px 18px;
  border-radius:10px;font-size:14px;font-weight:600;cursor:pointer;
  white-space:nowrap;transition:background .2s}}
.btn:hover{{background:#243d6b}}
.btn:disabled{{background:#9ca3af;cursor:default}}
/* ── Session bar ── */
.sbar{{background:#fff;padding:7px 14px;border-bottom:1px solid #e5e7eb;
  display:flex;align-items:center;justify-content:space-between;
  font-size:12px;color:#6b7280;flex-shrink:0}}
.sbar-name{{font-weight:600;color:#1a2b4a;font-size:13px}}
.btn-end{{background:#ef4444;color:#fff;border:none;padding:5px 11px;
  border-radius:6px;font-size:11px;font-weight:600;cursor:pointer}}
/* ── Messages ── */
.messages{{flex:1;overflow-y:auto;padding:14px;display:flex;
  flex-direction:column;gap:10px}}
.msg{{display:flex;flex-direction:column;max-width:80%}}
.msg.user{{align-self:flex-end;align-items:flex-end}}
.msg.assistant{{align-self:flex-start;align-items:flex-start}}
.bubble{{padding:9px 13px;border-radius:16px;font-size:14px;
  line-height:1.55;white-space:pre-wrap}}
.msg.user .bubble{{background:#1a2b4a;color:#fff;border-bottom-right-radius:4px}}
.msg.assistant .bubble{{background:#fff;color:#111827;
  border-bottom-left-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
/* typing dots */
.dots{{display:flex;gap:4px;padding:4px 2px}}
.dot{{width:7px;height:7px;background:#9ca3af;border-radius:50%;
  animation:bounce 1.4s infinite ease-in-out both}}
.dot:nth-child(1){{animation-delay:-.32s}}
.dot:nth-child(2){{animation-delay:-.16s}}
@keyframes bounce{{0%,80%,100%{{transform:scale(.6)}}40%{{transform:scale(1)}}}}
/* ── Input bar ── */
.ibar{{background:#fff;border-top:1px solid #e5e7eb;padding:10px 14px;
  display:flex;gap:8px;flex-shrink:0}}
.ibar input{{flex:1;padding:9px 13px;border:1.5px solid #d1d5db;
  border-radius:24px;font-size:14px;outline:none}}
.ibar input:focus{{border-color:#1a2b4a}}
.btn-send{{background:#1a2b4a;color:#fff;border:none;width:38px;height:38px;
  border-radius:50%;display:flex;align-items:center;justify-content:center;
  cursor:pointer;flex-shrink:0}}
.btn-send:disabled{{background:#d1d5db;cursor:default}}
/* ── Summary ── */
#summary-screen{{overflow-y:auto;padding:20px}}
.sum-card{{background:#fff;border-radius:12px;padding:22px;
  max-width:500px;margin:0 auto;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
.sum-title{{font-size:17px;font-weight:700;color:#1a2b4a;margin-bottom:14px}}
.sum-sec{{margin-bottom:14px}}
.sum-lbl{{font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.5px;color:#6b7280;margin-bottom:5px}}
.sum-txt{{font-size:13px;color:#374151;line-height:1.55}}
.kr-row{{display:flex;align-items:center;gap:7px;padding:5px 0;font-size:13px}}
.kr-dot{{width:9px;height:9px;border-radius:50%;flex-shrink:0}}
.kr-dot.green{{background:#10b981}}.kr-dot.yellow{{background:#f59e0b}}
.kr-dot.red{{background:#ef4444}}
.kr-pct{{font-weight:600;color:#1a2b4a;margin-left:auto}}
.btn-restart{{margin-top:16px;width:100%;padding:11px;background:#1a2b4a;
  color:#fff;border:none;border-radius:10px;font-size:14px;
  font-weight:600;cursor:pointer}}
.cta-link{{display:block;text-align:center;margin-top:10px;color:#1a2b4a;
  font-size:13px;font-weight:500;text-decoration:none}}
/* ── Toast ── */
.toast{{position:fixed;bottom:72px;left:50%;transform:translateX(-50%);
  background:#ef4444;color:#fff;padding:9px 16px;border-radius:8px;
  font-size:13px;display:none;z-index:99}}
</style>
</head>
<body>

<div class="hdr">
  <div>
    <div class="hdr-logo">ABN Consulting</div>
    <div class="hdr-sub">AI Co-Navigator</div>
  </div>
  <span class="hdr-badge">DEMO</span>
</div>

<!-- Landing -->
<div id="landing">
  <div class="lnd">
    <div class="lnd-icon">🧭</div>
    <div class="lnd-title">Your AI Co-Navigator</div>
    <div class="lnd-desc">
      Experience a live coaching session — tracking OKRs, navigating obstacles,
      and keeping your change journey on course.
    </div>
    <div class="name-row">
      <input type="text" id="name-in" placeholder="Your name to begin…" maxlength="80" />
      <button class="btn" id="start-btn" onclick="startSession()">Start →</button>
    </div>
  </div>
</div>

<!-- Chat -->
<div id="chat-screen">
  <div class="sbar">
    <span>Chatting as <span class="sbar-name" id="sbar-name"></span></span>
    <button class="btn-end" onclick="endSession()">End Session</button>
  </div>
  <div class="messages" id="msgs"></div>
  <div class="ibar">
    <input type="text" id="msg-in" placeholder="Type your message…"
           onkeydown="if(event.key==='Enter'&&!event.shiftKey){{event.preventDefault();sendMsg()}}" />
    <button class="btn-send" id="send-btn" onclick="sendMsg()">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.5">
        <line x1="22" y1="2" x2="11" y2="13"/>
        <polygon points="22 2 15 22 11 13 2 9 22 2"/>
      </svg>
    </button>
  </div>
</div>

<!-- Summary -->
<div id="summary-screen">
  <div class="sum-card">
    <div class="sum-title">Session Complete ✅</div>
    <div id="sum-content"></div>
    <button class="btn-restart" onclick="restart()">Start a New Session</button>
    <a class="cta-link" id="cta" href="{calendly_url}" target="_blank">
      📅 Book a real session with {coach_name}
    </a>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const API = "{api_base}";
const KEY = "{demo_key}";
let sid = null;

function toast(msg) {{
  const t = document.getElementById("toast");
  t.textContent = msg; t.style.display = "block";
  setTimeout(() => t.style.display = "none", 3500);
}}

async function api(path, method, body) {{
  const r = await fetch(API + path, {{
    method,
    headers: {{"Content-Type":"application/json","X-Demo-Key":KEY}},
    body: body ? JSON.stringify(body) : undefined
  }});
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}}

async function startSession() {{
  const name = document.getElementById("name-in").value.trim();
  if (!name) {{ document.getElementById("name-in").focus(); return; }}
  document.getElementById("start-btn").disabled = true;
  try {{
    const d = await api("/demo/session/start", "POST", {{name}});
    sid = d.session_id;
    document.getElementById("sbar-name").textContent = name;
    document.getElementById("landing").style.display = "none";
    document.getElementById("chat-screen").style.display = "flex";
    addMsg("assistant", d.message);
    document.getElementById("msg-in").focus();
  }} catch(e) {{
    toast("Couldn't start session — please try again.");
    document.getElementById("start-btn").disabled = false;
  }}
}}

function addMsg(role, text) {{
  const c = document.getElementById("msgs");
  const d = document.createElement("div");
  d.className = "msg " + role;
  d.innerHTML = `<div class="bubble">${{text}}</div>`;
  c.appendChild(d); c.scrollTop = c.scrollHeight;
}}

function showTyping() {{
  const c = document.getElementById("msgs");
  const d = document.createElement("div");
  d.id = "typing"; d.className = "msg assistant";
  d.innerHTML = `<div class="bubble"><div class="dots">
    <div class="dot"></div><div class="dot"></div><div class="dot"></div>
  </div></div>`;
  c.appendChild(d); c.scrollTop = c.scrollHeight;
}}

async function sendMsg() {{
  const inp = document.getElementById("msg-in");
  const text = inp.value.trim();
  if (!text || !sid) return;
  inp.value = "";
  document.getElementById("send-btn").disabled = true;
  addMsg("user", text);
  showTyping();
  try {{
    const d = await api(`/demo/session/${{sid}}/message`, "POST", {{message: text}});
    document.getElementById("typing")?.remove();
    addMsg("assistant", d.reply);
  }} catch(e) {{
    document.getElementById("typing")?.remove();
    toast("Couldn't send — please try again.");
  }} finally {{
    document.getElementById("send-btn").disabled = false;
    inp.focus();
  }}
}}

async function endSession() {{
  if (!confirm("End this session and receive your summary?")) return;
  document.querySelector(".btn-end").disabled = true;
  try {{
    const d = await api(`/demo/session/${{sid}}/end`, "POST", {{}});
    renderSummary(d);
    document.getElementById("chat-screen").style.display = "none";
    document.getElementById("summary-screen").style.display = "flex";
  }} catch(e) {{
    toast("Couldn't end session — please try again.");
    document.querySelector(".btn-end").disabled = false;
  }}
}}

function renderSummary(s) {{
  const log = s.weekly_log || {{}};
  let html = "";
  if (log.focus_goal)
    html += `<div class="sum-sec"><div class="sum-lbl">This Week's Focus</div>
      <div class="sum-txt">${{log.focus_goal}}</div></div>`;
  if (log.key_results?.length) {{
    html += `<div class="sum-sec"><div class="sum-lbl">Key Results</div>`;
    for (const kr of log.key_results) {{
      const c = kr.status_color || "green";
      html += `<div class="kr-row"><div class="kr-dot ${{c}}"></div>
        <span>${{kr.description}}</span>
        <span class="kr-pct">${{kr.status_pct}}%</span></div>`;
    }}
    html += `</div>`;
  }}
  const open = (log.obstacles||[]).filter(o => !o.resolved);
  if (open.length) {{
    html += `<div class="sum-sec"><div class="sum-lbl">Open Obstacles</div>`;
    for (const o of open)
      html += `<div class="sum-txt">⚠️ ${{o.description}}</div>`;
    html += `</div>`;
  }}
  if (log.mood_indicator) {{
    const m = ["😔","😐","🙂","😊","🌟"][log.mood_indicator-1];
    html += `<div class="sum-sec"><div class="sum-lbl">Mood</div>
      <div class="sum-txt">${{m}} ${{log.mood_indicator}}/5</div></div>`;
  }}
  if (s.summary_for_coach)
    html += `<div class="sum-sec"><div class="sum-lbl">Coach Notes</div>
      <div class="sum-txt">${{s.summary_for_coach}}</div></div>`;
  document.getElementById("sum-content").innerHTML = html;
}}

function restart() {{
  sid = null;
  document.getElementById("name-in").value = "";
  document.getElementById("msgs").innerHTML = "";
  document.getElementById("summary-screen").style.display = "none";
  document.getElementById("landing").style.display = "flex";
  document.getElementById("start-btn").disabled = false;
}}

document.getElementById("name-in").addEventListener("keydown",
  e => {{ if (e.key === "Enter") startSession(); }});
</script>
</body>
</html>"""
