// Delegation UX screens, revision 3 (review fixes + panel answer, two-expert turn, mobile Home and Work tab, right-panel sheet). Page: feat/delegate-task-ui.
import { writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { C, F, div, row, col, text, muted, sec, esc, icon, badge, dot, button, iconButton, card, divider, chip, avatarFor, page1440, page390, fab, userMessage, assistantText, composer, inputSmall, dockBar, toolChain, skeletonBlock, kitDir } from "./delegate-lib.mjs";

const out = join(kitDir, "delegate"); mkdirSync(out, { recursive: true });
const entries = [];
function emit(id, name, html, width, height) { writeFileSync(join(out, id + ".html"), html); entries.push({ id, title: "Delegate/" + name, name, page: "feat/delegate-task-ui", artboardName: name, file: "delegate/" + id + ".html", width, height, bytes: html.length, exact: true }); console.log("ok", name, html.length); }

const ASK = "Draft the PRD for the onboarding revamp and get me an eng estimate. Use the research in Notion.";
const BRIEF = "Draft the PRD for the onboarding revamp: goals, scope, acceptance criteria. Use the Notion research notes and last week's funnel numbers.";

// ---------- Status helpers ----------
const STATUS = { proposed: ["warning", "Waiting for you"], approved: ["success", "Approved"], queued: ["info", "Queued"], working: ["accent", "Working"], "needs-input": ["warning", "Needs you"], done: ["success", "Done"], failed: ["error", "Failed"] };
const stBadge = (s) => badge(STATUS[s][0], STATUS[s][1]);
const DOT = { warning: C.warnDot, info: C.z400, accent: C.accent, success: C.successDot, error: C.errDot };
const metaItem = (ic, label, value) => row(6, icon(ic, 14, C.z500) + muted(F.small, label) + text(F.smallM, value));
const cell = (l, v) => col(2, muted(F.small, l) + text(F.bodyM, v), "flex:1;min-width:0");

// ---------- Chain nodes: cards that sit on the timeline like a GitHub comment ----------
export function approvalNode({ width = 806, compact = false } = {}) {
  const line = div(`display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid ${C.z100};background-color:${C.z50}`, avatarFor("Alex", 28) + div(`${F.body};flex:1;min-width:0`, "Hand off " + '<span style="font-weight:500">“Onboarding revamp PRD”</span>' + " to Alex · Product Manager") + muted(F.small, "Ask First"));
  const details = div(`display:flex;flex-direction:column;gap:12px;padding:14px 16px`,
    col(4, text(F.eyebrow, "Why Alex") + text(F.body, "Product scoping work, and Alex already holds last week's funnel notes.")) +
    col(4, text(F.eyebrow, "Brief") + text(F.body, BRIEF)) +
    div(`display:flex;gap:16px;padding-top:10px;border-top:1px solid ${C.z100}`, cell("Expected back", "PRD draft as a file") + cell("By", "Today 6:00 pm") + cell("Cap", "$2.00 of $5 weekly") + cell("Owner", "You")));
  const actions = div(`display:flex;align-items:center;gap:8px;padding:12px 16px;border-top:1px solid ${C.z100}${compact ? ";flex-wrap:wrap" : ""}`, button("primary", "Approve", { size: "small", leadingIcon: "Tick02Icon", extra: "min-width:0" }) + button("secondary", "Reject", { size: "small", extra: "min-width:0" }) + button("ghost", "Edit brief", { size: "small", leadingIcon: "Edit02Icon" }) + button("ghost", "Pick another expert", { size: "small" }) + (compact ? "" : div("flex:1") + row(6, icon("ShieldKeyIcon", 14, C.z500) + div(`${F.small};color:${C.z600};text-decoration-line:underline`, "Always allow hand-offs to Alex"))));
  return div(`display:flex;flex-direction:column;width:${width}px;border-radius:12px;border:1px solid ${C.z200};background-color:#FEFEFE;overflow:hidden;box-sizing:border-box`, line + details + actions);
}
export function questionNode({ width = 806, compact = false, inPanel = false } = {}) {
  const line = div(`display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid ${C.z100};background-color:${C.warnBg}`, avatarFor("Alex", 28) + div(`${F.body};flex:1;min-width:0`, '<span style="font-weight:500">Alex asks</span>' + " · paused on “Onboarding revamp PRD” · 3m") + badge("warning", "Needs you"));
  const dim = inPanel ? `color:${C.z400};border-color:${C.z100};box-shadow:none` : "";
  const chips = row(8, ["Q4 release", "December mini-launch", "Both, mark the difference"].map((l) => button("secondary", l, { size: "xs", extra: dim })).join(""), "flex-wrap:wrap");
  // While the answer is being typed in the right panel the card hands over: no input, one line that points at the panel.
  const answer = inPanel
    ? div(`display:flex;align-items:center;gap:8px;height:36px;padding:0 12px;border-radius:12px;border:1px dashed ${C.z200};background-color:${C.z50};box-sizing:border-box`, icon("LayoutRightIcon", 14, C.z500) + div(`${F.small};color:${C.z600};white-space:nowrap`, "You're answering this in the panel") + div("flex:1") + icon("ArrowRight01Icon", 14, C.z400))
    : row(8, inputSmall("Or answer in your own words…") + button("primary", "Send", { size: "small", extra: "min-width:0;height:36px" }));
  const body = div(`display:flex;flex-direction:column;gap:10px;padding:14px 16px`, text(F.body, "Should the PRD target the Q4 release train or the December mini-launch? The funnel notes mention both and the scope changes a lot.") + chips + answer);
  const actions = div(`display:flex;align-items:center;gap:8px;padding:10px 16px;border-top:1px solid ${C.z100}`, button("ghost", "Open Alex's thread", { size: "small", rightIcon: "ArrowRight01Icon" }) + button("ghost", "Take over", { size: "small", leadingIcon: "UserSwitchIcon" }) + (compact ? "" : div("flex:1") + muted(F.small, inPanel ? "Same answer wherever you send it" : "Your answer continues Alex's thread")));
  return div(`display:flex;flex-direction:column;width:${width}px;border-radius:12px;border:1px solid rgba(245,158,11,0.35);background-color:#FEFEFE;overflow:hidden;box-sizing:border-box`, line + body + actions);
}
export function skeletonNode(width = 806) {
  const line = (w) => skeletonBlock(w, 12);
  return div(`display:flex;flex-direction:column;width:${width}px;border-radius:12px;border:1px solid ${C.z200};background-color:#FEFEFE;overflow:hidden;box-sizing:border-box`, div(`display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid ${C.z100};background-color:${C.z50}`, skeletonBlock("28px", 28, "border-radius:9999px") + line("260px")) + div("display:flex;flex-direction:column;gap:10px;padding:14px 16px", line("48px") + line("92%") + line("70%") + row(16, line("140px") + line("110px") + line("160px"))) + div(`display:flex;gap:8px;padding:12px 16px;border-top:1px solid ${C.z100}`, skeletonBlock("104px", 36, "border-radius:9999px") + skeletonBlock("88px", 36, "border-radius:9999px")));
}
// One line under the chain once the hand-off is running: loader, count, timing, arrow → right panel.
export function statusLine(state, { width = 806, elapsed, label, detail } = {}) {
  const cfg = {
    queued: ["pending", "1 expert queued", "Alex starts in about 1 minute"],
    working: ["spinner", "1 expert working", "Alex · " + (elapsed || "2m 14s") + " · $0.12"],
    answered: ["spinner", "1 expert working", "Alex · resumed · " + (elapsed || "4m 10s") + " · $0.21"],
    done: ["done", "Alex reported back", (elapsed || "6m 40s") + " · $0.31 · 1 file"],
    failed: ["error", "Alex stopped", "budget cap · " + (elapsed || "4m 10s") + " · $5.00"],
  }[state];
  if (label) cfg[1] = label;
  if (detail) cfg[2] = detail;
  const ic = cfg[0] === "spinner" ? icon("Loading03Icon", 16, "#A855F7") : cfg[0] === "done" ? icon("CheckmarkCircle02Icon", 16, C.successText) : cfg[0] === "error" ? icon("Alert02Icon", 16, C.errText) : icon("CircleIcon", 16, C.z400);
  const right = state === "failed" ? row(8, button("secondary", "Retry", { size: "xs", leadingIcon: "ArrowReloadHorizontalIcon" }) + icon("ArrowRight01Icon", 16, C.z500)) : row(6, div(`${F.small};color:${C.z600}`, "Open") + icon("ArrowRight01Icon", 16, C.z500));
  return div(`display:flex;align-items:center;gap:10px;width:${width}px;height:40px;padding:0 12px 0 8px;border-radius:10px;box-sizing:border-box`, ic + div(`${F.bodyM};color:${C.z800};white-space:nowrap`, cfg[1]) + div(`${F.body};color:${C.z500};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0`, "· " + cfg[2]) + div("flex:1") + right);
}

// ---------- Otto's turn: the timeline ----------
function ottoTurn(state, { width = 806, mode = "ask", compact = false } = {}) {
  const base = [{ icon: "UserGroupIcon", text: "Checked who on the team is free", state: "done" }, { icon: "Notebook01Icon", text: 'Read "Onboarding research" in Notion', state: "done" }];
  const handoff = (st, extra = {}) => ({ icon: "Robot01Icon", text: "Handing off to a teammate: Alex", state: st, ...extra });
  const approvedRow = { icon: "Tick02Icon", iconColor: C.successText, text: "You approved the hand-off to Alex · 10:42", state: "done" };
  const answeredRow = { icon: "Tick02Icon", iconColor: C.successText, text: "You answered: Q4 release train · 10:46", state: "done" };
  const settled = (label) => toolChain([], { collapsed: label });
  const heading = mode === "ask" ? "Checked the team, read Notion, handed off to Alex, you approved" : mode === "auto" ? "Checked the team, read Notion, handed off to Alex (auto)" : "Checked the team, read Notion, handed off to Alex";
  let chain, status = "", after = "";
  switch (state) {
    case "skeleton": chain = toolChain([...base, handoff("running", { text: "Handing off to a teammate…" }), { node: skeletonNode(width + 16) }], { width }); break;
    case "proposed": chain = toolChain([...base, handoff("running", { tag: "Waiting for you" }), { node: approvalNode({ width: width + 16, compact }) }], { width }); after = "Two pieces of work: a PRD and an estimate. I'd hand the PRD to Alex and queue Devon for the estimate. Approve and I'll start, or edit the brief first."; break;
    case "approved": chain = toolChain([...base, handoff("done"), approvedRow], { width }); status = statusLine("queued", { width }); after = "Approved. Alex picks it up as soon as the current routine finishes; I'll bring the draft back here."; break;
    case "working": chain = settled(heading); status = statusLine("working", { width }); after = "Alex is on the PRD. I'll queue Devon for the estimate once the draft lands and keep the result here."; break;
    case "needs-input": chain = toolChain([...base, handoff("done"), approvedRow, { icon: "MessageQuestionIcon", text: "Alex asked a question", state: "running", tag: "Needs you" }, { node: questionNode({ width: width + 16, compact }) }], { width }); after = "Alex needs one call from you before continuing."; break;
    case "answered": chain = settled("Checked the team, handed off to Alex, you approved, Alex asked, you answered"); status = statusLine("answered", { width }); after = "Thanks, Q4 it is. Alex is back on the draft."; break;
    case "done": chain = settled(heading); status = statusLine("done", { width }); after = "Alex's draft is in. Three open questions need your call before Devon can estimate; want me to walk you through them?"; break;
    case "failed": chain = settled(heading); status = statusLine("failed", { width }); after = "Alex ran out of weekly budget mid-draft. Nothing was sent. I can raise the cap by $2 and let Alex finish, retry as is, or you take the partial draft."; break;
    // The question is being answered from the right panel: the card stays on the wire but hands over to the panel.
    case "panel-answer": chain = toolChain([...base, handoff("done"), approvedRow, { icon: "MessageQuestionIcon", text: "Alex asked a question", state: "running", tag: "Needs you" }, { node: questionNode({ width: width + 16, compact, inPanel: true }) }], { width }); after = "Alex needs one call from you before continuing."; break;
    // Afterwards the card is gone: the chain gets a "You answered from the panel" row and the status line returns.
    case "panel-answered": chain = toolChain([...base, handoff("done"), approvedRow, { icon: "MessageQuestionIcon", text: "Alex asked a question", state: "done" }, { icon: "Tick02Icon", iconColor: C.successText, text: "You answered from the panel: Q4 release train · 10:46", state: "done" }], { width }); status = statusLine("answered", { width, detail: "Alex · resumed after your answer · 4m 10s · $0.21" }); after = "Thanks, Q4 it is. Alex is back on the draft."; break;
    // Two hand-offs in one turn: one approval row each, then one status line per expert.
    case "multi": chain = toolChain([...base, handoff("done"), approvedRow, handoff("done", { text: "Handing off to a teammate: Devon" }), { icon: "Tick02Icon", iconColor: C.successText, text: "You approved the hand-off to Devon · 10:42", state: "done" }], { width }); status = col(0, statusLine("working", { width, label: "Alex working", detail: "PRD draft · scope section · 2m 14s · $0.12" }) + statusLine("working", { width, label: "Devon working", detail: "Eng estimate · reading the brief · 1m 02s · $0.05" })); after = "Alex is drafting the PRD and Devon is estimating from the same brief in parallel. I'll bring both back here and reconcile them."; break;
  }
  return col(8, chain + status + (after ? div("padding-top:4px", assistantText([after], width)) : ""));
}

// ---------- Docked summary above the composer: one line, click → right panel ----------
function dockFor(state, width = 806) {
  const cfg = { skeleton: ["spinner", "Handing off to Alex…"], proposed: ["warn", "1 hand-off waiting for your approval"], approved: ["pending", "1 expert queued"], queued: ["pending", "1 expert queued"], working: ["spinner", "1 expert working · 1 queued"], "needs-input": ["warn", "1 expert needs you"], answered: ["spinner", "1 expert working · 1 queued"], "panel-answer": ["warn", "1 expert needs you"], "panel-answered": ["spinner", "1 expert working · 1 queued"], multi: ["spinner", "2 experts working"], done: ["spinner", "1 expert working · 1 done"], failed: ["error", "1 expert stopped"] }[state];
  return dockBar({ icon: cfg[0], title: cfg[1], count: "", width, open: true });
}

// ---------- Right panel: Work tab with list and detail ----------
function treeNode({ who, role, state, line, elapsed, spend }) {
  const s = STATUS[state];
  return div(`display:flex;gap:10px;padding:10px 12px;border-radius:12px;background-color:${state === "needs-input" ? C.warnBg : "#FEFEFE"};border:1px solid ${state === "needs-input" ? "rgba(245,158,11,0.25)" : C.z200};box-sizing:border-box;width:100%`,
    avatarFor(who, 28) + col(4, div("display:flex;align-items:center;justify-content:space-between;gap:8px", row(6, text(F.bodyM, who) + sec(F.small, role)) + row(6, dot(DOT[s[0]], 6) + text(F.smallM, s[1]))) + sec(F.small, line) + row(10, muted(F.small, elapsed) + muted(F.small, spend) + div("flex:1") + icon("ArrowRight01Icon", 14, C.z400)), "flex:1;min-width:0"));
}
const activityRow = (kind, t, time) => { const m = { answer: ["Comment01Icon", C.z600, "You answered"], thought: ["AiBrain01Icon", C.z400, "Thought"], action: ["ZapIcon", C.z600, "Action"], question: ["MessageQuestionIcon", C.warnText, "Question"], response: ["CheckmarkCircle02Icon", C.successText, "Response"], error: ["Alert02Icon", C.errText, "Error"], approval: ["Tick02Icon", C.successText, "Approved"] }[kind]; return div(`display:flex;gap:12px;padding:6px 0`, div(`display:flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:9999px;background-color:${C.z50};border:1px solid ${C.z200};flex-shrink:0`, icon(m[0], 14, m[1])) + col(2, row(8, text(F.smallM, m[2]) + muted(F.small, time)) + text(F.body, t), "flex:1")); };
function detailView(state = "done") {
  const receipt = (t, sub) => row(10, icon("CheckmarkCircle02Icon", 16, C.successText) + col(0, text(F.body, t) + (sub ? muted(F.small, sub) : "")));
  const timeline = {
    queued: [["approval", "You approved the hand-off · $2.00 cap", "10:42"]],
    working: [["approval", "You approved the hand-off · $2.00 cap", "10:42"], ["action", "Read 3 Notion pages: research, funnel, interviews", "10:42"], ["action", "Pulled last week's activation numbers", "10:43"], ["thought", "Writing the scope section…", "10:44"]],
    "needs-input": [["approval", "You approved the hand-off · $2.00 cap", "10:42"], ["action", "Read research and funnel sources", "10:42"], ["question", "Asked you: Q4 release train or December mini-launch?", "10:44"]],
    "panel-answer": [["approval", "You approved the hand-off · $2.00 cap", "10:42"], ["action", "Read research and funnel sources", "10:42"], ["question", "Asked you: Q4 release train or December mini-launch?", "10:44"]],
    answered: [["approval", "You approved the hand-off · $2.00 cap", "10:42"], ["action", "Read research and funnel sources", "10:42"], ["question", "Asked you: Q4 release train or December mini-launch?", "10:44"], ["answer", "Q4 release train. Keep December as a stretch note.", "10:46"], ["thought", "Resumed: writing the scope section for Q4…", "10:46"]],
    done: [["approval", "You approved the hand-off · $2.00 cap", "10:42"], ["action", "Read research and funnel sources", "10:42"], ["question", "Asked you: Q4 or December? You answered Q4.", "10:44"], ["response", "Reported back to Otto with 1 file", "10:48"]],
    failed: [["approval", "You approved the hand-off · $2.00 cap", "10:42"], ["action", "Read research and funnel sources", "10:42"], ["error", "Weekly budget cap reached ($5.00). Draft saved up to the scope section.", "10:46"]],
  }[state] || [];
  const stateBlock = state === "working" ? col(8, text(F.eyebrow, "Controls") + div("display:flex;gap:8px;flex-wrap:wrap", button("secondary", "Open thread", { size: "xs", rightIcon: "ArrowRight01Icon" }) + button("secondary", "Nudge", { size: "xs", leadingIcon: "Comment01Icon" }) + button("secondary", "Pause", { size: "xs", leadingIcon: "PauseIcon" })))
    : state === "panel-answer" ? col(8, text(F.eyebrow, "Alex asks") + div(`display:flex;flex-direction:column;gap:8px;padding:12px;border-radius:12px;background-color:${C.warnBg};border:1px solid rgba(245,158,11,0.25)`, text(F.body, "Q4 release train or December mini-launch?") + row(6, button("primary", "Q4", { size: "xs", leadingIcon: "Tick02Icon" }) + button("secondary", "December", { size: "xs" }) + button("secondary", "Both", { size: "xs" }), "flex-wrap:wrap") + div(`display:flex;align-items:center;height:36px;padding:0 16px;border-radius:12px;border:1px solid ${C.z800};background-color:#FEFEFE;${F.body};box-sizing:border-box;white-space:nowrap;overflow:hidden`, esc("Keep December as a stretch note.") + div(`width:1px;height:16px;background-color:${C.black};margin-left:1px`)) + row(8, button("primary", "Send to Alex", { size: "small", rightIcon: "ArrowRight01Icon", extra: "min-width:0" }) + muted(F.small, "Closes the card in the chat"))))
    : state === "answered" ? col(8, text(F.eyebrow, "Your answer") + div(`display:flex;gap:10px;padding:12px;border-radius:12px;background-color:${C.z50};border:1px solid ${C.z200}`, icon("CheckmarkCircle02Icon", 16, C.successText) + col(2, text(F.body, "Q4 release train. Keep December as a stretch note.") + muted(F.small, "Sent 10:46 from this panel · Alex resumed"), "flex:1;min-width:0")) + div("display:flex;gap:8px;flex-wrap:wrap", button("secondary", "Open thread", { size: "xs", rightIcon: "ArrowRight01Icon" }) + button("secondary", "Nudge", { size: "xs", leadingIcon: "Comment01Icon" }) + button("secondary", "Pause", { size: "xs", leadingIcon: "PauseIcon" })))
    : state === "needs-input" ? col(8, text(F.eyebrow, "Alex asks") + div(`display:flex;flex-direction:column;gap:8px;padding:12px;border-radius:12px;background-color:${C.warnBg};border:1px solid rgba(245,158,11,0.25)`, text(F.body, "Q4 release train or December mini-launch?") + row(6, button("secondary", "Q4", { size: "xs" }) + button("secondary", "December", { size: "xs" }) + button("secondary", "Both", { size: "xs" }), "flex-wrap:wrap") + inputSmall("Answer…")))
    : state === "done" ? col(8, text(F.eyebrow, "What came back") + receipt("PRD draft with 9 acceptance criteria", "PRD-onboarding-revamp-v1.md · 12 KB") + receipt("Open questions · 3 need your call") + receipt("Funnel numbers labelled FACT / INFERENCE / UNKNOWN") + div("display:flex;gap:8px;flex-wrap:wrap;padding-top:4px", button("primary", "Open thread", { size: "xs", rightIcon: "ArrowRight01Icon" }) + button("secondary", "Re-delegate", { size: "xs", leadingIcon: "ArrowReloadHorizontalIcon" })))
    : state === "failed" ? col(8, text(F.eyebrow, "What to do") + div("display:flex;gap:8px;flex-wrap:wrap", button("primary", "Raise budget and retry", { size: "xs", leadingIcon: "Wallet01Icon" }) + button("secondary", "Retry", { size: "xs" }) + button("secondary", "Do it myself", { size: "xs" })))
    : col(8, text(F.eyebrow, "Status") + sec(F.body, "Alex is finishing a scheduled routine. Starts in about 1 minute.") + button("secondary", "Cancel", { size: "xs" }));
  const sub = { queued: "Approved 10:42 · starts soon", working: "Working · 2m 14s · $0.12", "needs-input": "Paused with a question · 3m 02s", "panel-answer": "Paused with a question · 3m 02s", answered: "Resumed 10:46 · 4m 10s · $0.21", done: "Today 10:41 → 10:48 · 6m 40s · $0.31", failed: "Stopped 10:46 · $5.00" }[state];
  return col(16,
    row(8, icon("ArrowLeft01Icon", 14, C.z500) + sec(F.small, "All work")) +
    div("display:flex;align-items:flex-start;justify-content:space-between;gap:8px", row(10, avatarFor("Otto", 24) + icon("ArrowRight01Icon", 12, C.z400) + avatarFor("Alex", 24) + col(0, text(F.h5, "Onboarding revamp PRD") + muted(F.small, sub))) + stBadge(state === "panel-answer" ? "needs-input" : state === "answered" ? "working" : state)) +
    stateBlock +
    col(6, text(F.eyebrow, "Brief") + text(F.body, BRIEF)) +
    div("display:flex;gap:8px", cell("Owner", "You") + cell("Expected back", "PRD file") + cell("Cap", "$2.00")) +
    col(6, text(F.eyebrow, "Timeline") + col(0, timeline.map(([k, t, tm]) => activityRow(k, t, tm)).join(""))), "padding:16px");
}
export function workPanel(nodes, { width = 384, height = 900, tab = "work", waiting = 0, empty = false, detail = null, spent = "$0.31" } = {}) {
  const tabs = row(0, [["Files", "Folder01Icon", tab === "files"], ["Work", "Task01Icon", tab === "work"]].map(([l, ic, on]) => div(`display:flex;align-items:center;gap:6px;height:40px;padding:0 4px;margin-right:16px;border-bottom:2px solid ${on ? C.black : "transparent"};${on ? F.bodyM : F.body};color:${on ? C.black : C.z500}`, icon(ic, 16) + esc(l) + (l === "Work" && waiting ? badge("warning", String(waiting), "small") : ""))).join(""));
  const header = div(`display:flex;align-items:center;justify-content:space-between;padding:12px 16px 0 16px;border-bottom:1px solid ${C.z200}`, tabs + iconButton("ArrowExpand01Icon", { size: 28, extra: "border:none;box-shadow:none;background-color:transparent" }));
  let body;
  if (detail) body = detailView(detail);
  else if (empty) body = div(`display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;padding:48px 24px;text-align:center`, icon("UserGroupIcon", 24, C.z400) + text(F.h5, "Nothing delegated yet") + sec(F.body, "When Otto hands work to an expert, it shows up here with its status, cost and controls."));
  else body = col(16, (waiting ? col(8, text(F.eyebrow, "Waiting for you") + nodes.filter((n) => n.state === "needs-input").map(treeNode).join("")) : "") + col(8, div("display:flex;align-items:center;justify-content:space-between", text(F.eyebrow, "This chat") + muted(F.small, "Otto → " + nodes.length + " experts")) + col(6, nodes.filter((n) => !(waiting && n.state === "needs-input")).map(treeNode).join(""))) + col(8, text(F.eyebrow, "Totals") + div(`display:flex;gap:8px`, [["Working", String(nodes.filter((n) => n.state === "working").length)], ["Waiting", String(waiting)], ["Spent", spent], ["Cap", "$4.00"]].map(([l, v]) => div(`display:flex;flex-direction:column;gap:2px;flex:1;min-width:0;padding:10px 12px;border-radius:12px;background-color:${C.z50}`, muted(F.small, l, "white-space:nowrap") + text(F.h5, v))).join(""))), "padding:16px");
  return div(`display:flex;flex-direction:column;width:${width}px;height:${height}px;background-color:#FEFEFE;border-left:1px solid ${C.z200};box-sizing:border-box;flex-shrink:0;overflow:hidden`, header + body);
}

// ---------- Copilot screen composition ----------
const NODES = {
  proposed: [], skeleton: [], approved: [{ who: "Alex", role: "Product Manager", state: "queued", line: "Starts after the current routine", elapsed: "—", spend: "$0.00" }],
  working: [{ who: "Alex", role: "Product Manager", state: "working", line: "Writing the scope section", elapsed: "2m 14s", spend: "$0.12" }, { who: "Devon", role: "Application Security Engineer", state: "queued", line: "Eng estimate · after the draft", elapsed: "—", spend: "$0.00" }],
  "needs-input": [{ who: "Alex", role: "Product Manager", state: "needs-input", line: "Q4 release or December mini-launch?", elapsed: "3m 02s", spend: "$0.19" }, { who: "Devon", role: "Application Security Engineer", state: "queued", line: "Eng estimate · after the draft", elapsed: "—", spend: "$0.00" }],
  done: [{ who: "Alex", role: "Product Manager", state: "done", line: "PRD draft returned · 1 file", elapsed: "6m 40s", spend: "$0.31" }, { who: "Devon", role: "Application Security Engineer", state: "working", line: "Estimating from the acceptance criteria", elapsed: "0m 48s", spend: "$0.04" }],
  failed: [{ who: "Alex", role: "Product Manager", state: "failed", line: "Weekly budget cap reached", elapsed: "4m 10s", spend: "$5.00" }, { who: "Devon", role: "Application Security Engineer", state: "queued", line: "Eng estimate · after the draft", elapsed: "—", spend: "$0.00" }],
};
NODES.answered = NODES.working;
NODES["panel-answer"] = NODES["needs-input"];
NODES["panel-answered"] = NODES.working;
NODES.multi = [{ who: "Alex", role: "Product Manager", state: "working", line: "PRD draft · writing the scope section", elapsed: "2m 14s", spend: "$0.12" }, { who: "Devon", role: "Application Security Engineer", state: "working", line: "Eng estimate · reading the brief", elapsed: "1m 02s", spend: "$0.05" }];
function copilotScreen({ state, mode = "ask", panel = true, detail = null }) {
  const chat = div(`display:flex;flex-direction:column;flex:1;min-width:0;align-items:center;position:relative`,
    div(`display:flex;flex-direction:column;gap:24px;width:806px;padding-top:56px;flex:1`, userMessage(ASK) + ottoTurn(state, { mode })) +
    div(`display:flex;flex-direction:column;width:806px;padding-bottom:24px`, dockFor(state, 806) + composer(806, "Reply to Otto…")));
  const nodes = NODES[state] || [];
  const waiting = state === "needs-input" || state === "panel-answer" ? 1 : 0;
  return page1440(div(`display:flex;flex-direction:row;height:900px;position:relative`, chat + (panel ? workPanel(nodes, { waiting, empty: nodes.length === 0 && !detail, detail, spent: state === "multi" ? "$0.17" : "$0.31" }) : "") + div(`position:absolute;right:${panel ? 400 : 16}px;top:12px`, iconButton("Folder01Icon", { size: 32, extra: "border:none;box-shadow:none;background-color:transparent" }))) + (panel ? "" : fab()));
}
emit("copilot-skeleton-1440", "delegate/copilot/skeleton/1440", copilotScreen({ state: "skeleton" }), 1440, 900);
emit("copilot-proposed-1440", "delegate/copilot/proposed/1440", copilotScreen({ state: "proposed" }), 1440, 900);
emit("copilot-approved-1440", "delegate/copilot/approved/1440", copilotScreen({ state: "approved", detail: "queued" }), 1440, 900);
emit("copilot-working-1440", "delegate/copilot/working/1440", copilotScreen({ state: "working" }), 1440, 900);
emit("copilot-working-panel-detail-1440", "delegate/copilot/working/panel-detail/1440", copilotScreen({ state: "working", detail: "working" }), 1440, 900);
emit("copilot-needs-input-1440", "delegate/copilot/needs-input/1440", copilotScreen({ state: "needs-input", detail: "needs-input" }), 1440, 900);
emit("copilot-answered-1440", "delegate/copilot/answered/1440", copilotScreen({ state: "answered" }), 1440, 900);
emit("copilot-done-1440", "delegate/copilot/done/1440", copilotScreen({ state: "done", detail: "done" }), 1440, 900);
emit("copilot-failed-1440", "delegate/copilot/failed/1440", copilotScreen({ state: "failed", detail: "failed" }), 1440, 900);
emit("copilot-auto-mode-1440", "delegate/copilot/auto-mode/1440", copilotScreen({ state: "working", mode: "auto" }), 1440, 900);
emit("copilot-unsupervised-1440", "delegate/copilot/unsupervised/1440", copilotScreen({ state: "working", mode: "unsupervised" }), 1440, 900);
emit("copilot-needs-input-panel-answer-1440", "delegate/copilot/needs-input/panel-answer/1440", copilotScreen({ state: "panel-answer", detail: "panel-answer" }), 1440, 900);
emit("copilot-needs-input-panel-answered-1440", "delegate/copilot/needs-input/panel-answered/1440", copilotScreen({ state: "panel-answered", detail: "answered" }), 1440, 900);
emit("copilot-multi-1440", "delegate/copilot/multi/1440", copilotScreen({ state: "multi" }), 1440, 900);

function copilot390(state, detail) {
  const w = 342;
  return page390(div(`display:flex;flex-direction:column;flex:1;align-items:center`, div(`display:flex;flex-direction:column;gap:20px;width:${w}px;padding-top:76px;flex:1`, userMessage(ASK, w) + ottoTurn(state, { width: w, compact: true })) + div(`display:flex;flex-direction:column;width:${w}px;padding-bottom:16px`, dockFor(state, w) + composer(w, "Reply…"))), { height: 1240 });
}
emit("copilot-proposed-390", "delegate/copilot/proposed/390", copilot390("proposed"), 390, 1240);
emit("copilot-working-390", "delegate/copilot/working/390", copilot390("working"), 390, 1240);
emit("copilot-needs-input-390", "delegate/copilot/needs-input/390", copilot390("needs-input"), 390, 1240);
emit("copilot-panel-detail-390", "delegate/copilot/panel-detail/390", div(`position:relative;width:390px;height:844px;overflow:hidden;background-color:${C.inset}`, div(`position:absolute;left:0px;top:0px;width:390px;height:844px;background-color:rgba(20,20,20,0.35)`) + div(`position:absolute;left:0px;top:80px;width:390px;height:764px;border-radius:24px 24px 0 0;overflow:hidden;background-color:#FEFEFE`, div(`display:flex;justify-content:center;padding-top:8px`, div(`width:36px;height:4px;border-radius:9999px;background-color:${C.z200}`)) + detailView("working"))), 390, 844);

// ---------- Component sheets ----------
emit("component-chain-nodes", "delegate/components/chain-nodes/all-states", div(`display:flex;flex-direction:column;gap:28px;padding:32px;width:900px;background-color:#FEFEFE`, [
  ["Approval node (ask-first) · attached to the wire", toolChain([{ icon: "Robot01Icon", text: "Handing off to a teammate: Alex", state: "running", tag: "Waiting for you" }, { node: approvalNode({ width: 852 }) }], { width: 836 })],
  ["Approved → row, then the chain closes like any other", toolChain([{ icon: "Robot01Icon", text: "Handing off to a teammate: Alex", state: "done" }, { icon: "Tick02Icon", iconColor: C.successText, text: "You approved the hand-off to Alex · 10:42", state: "done" }], { width: 836 })],
  ["Question node", toolChain([{ icon: "MessageQuestionIcon", text: "Alex asked a question", state: "running", tag: "Needs you" }, { node: questionNode({ width: 852 }) }], { width: 836 })],
  ["Loading node", toolChain([{ icon: "Robot01Icon", text: "Handing off to a teammate…", state: "running" }, { node: skeletonNode(852) }], { width: 836 })],
].map(([l, t]) => col(8, text(F.eyebrow, l) + t)).join("")), 900, 1400);
emit("component-status-line", "delegate/components/status-line/all-states", div(`display:flex;flex-direction:column;gap:20px;padding:32px;width:900px;background-color:#FEFEFE`, ["queued", "working", "answered", "done", "failed"].map((s) => col(6, text(F.eyebrow, s + " · sits under the closed chain · click opens the right panel") + statusLine(s, { width: 836 }))).join("")), 900, 620);
emit("component-dock-states", "delegate/components/delegation-dock/all-states", div(`display:flex;flex-direction:column;gap:24px;padding:32px;width:900px;background-color:${C.inset}`, [["Working", "working"], ["Needs you", "needs-input"], ["Waiting for approval", "proposed"], ["Failed", "failed"]].map(([l, s]) => col(6, text(F.eyebrow, l + " · click opens the right panel") + div("display:flex;flex-direction:column", dockFor(s, 806) + div(`height:24px;width:806px;border-radius:24px 24px 0 0;border:1px solid ${C.z200};background-color:#FEFEFE;position:relative`)))).join("") + col(6, text(F.eyebrow, "While a task list is running the task progress bar keeps the slot") + div("display:flex;flex-direction:column", dockBar({ icon: "spinner", title: "Writing the scope section", count: "2/4", width: 806 }) + div(`height:24px;width:806px;border-radius:24px 24px 0 0;border:1px solid ${C.z200};background-color:#FEFEFE;position:relative`)))), 900, 820);
emit("component-work-panel", "delegate/components/work-panel/states", div(`display:flex;gap:32px;padding:32px;background-color:${C.inset}`, [["List", workPanel(NODES.working, { height: 720 })], ["Detail · working", workPanel([], { detail: "working", height: 720 })], ["Detail · needs you", workPanel([], { detail: "needs-input", height: 720 })], ["Detail · done", workPanel([], { detail: "done", height: 720 })]].map(([l, p]) => col(8, text(F.eyebrow, l) + div(`border-radius:16px;overflow:hidden;border:1px solid ${C.z200}`, p))).join("")), 1760, 810);
emit("component-right-panel-detail", "delegate/components/right-panel-detail/all-states", div(`display:flex;flex-wrap:wrap;gap:32px;padding:32px;width:1720px;box-sizing:border-box;background-color:${C.inset}`, [["Queued", "queued"], ["Working", "working"], ["Needs you", "needs-input"], ["Answering from the panel", "panel-answer"], ["Answered · resumed", "answered"], ["Done", "done"], ["Failed", "failed"], ["List · two experts in one turn", null]].map(([l, d]) => col(8, text(F.eyebrow, l) + div(`border-radius:16px;overflow:hidden;border:1px solid ${C.z200}`, d ? workPanel([], { detail: d, height: 660 }) : workPanel(NODES.multi, { height: 660, spent: "$0.17" })))).join("")), 1720, 1470);
emit("component-tool-chain", "delegate/components/tool-chain/states", div(`display:flex;flex-direction:column;gap:24px;padding:32px;width:900px;background-color:#FEFEFE`, [
  ["Collapsed (settled turn)", toolChain([], { collapsed: "Checked the team, read Notion, handed off to Alex, you approved" })],
  ["Rows", toolChain([{ icon: "UserGroupIcon", text: "Checked who on the team is free", state: "done" }, { icon: "Robot01Icon", text: "Handing off to a teammate: Alex", state: "running", tag: "Waiting for you" }, { icon: "Tick02Icon", iconColor: C.successText, text: "You approved the hand-off to Alex · 10:42", state: "done" }, { icon: "Alert02Icon", text: "Alex stopped: weekly budget cap reached", state: "error" }])],
].map(([l, t]) => col(6, text(F.eyebrow, l) + t)).join("")), 900, 420);

// ---------- Expert thread ----------
const sentFrom = div(`display:inline-flex;align-items:center;gap:6px;height:24px;padding:0 8px;border-radius:9999px;background-color:${C.violet100};color:${C.violet700};${F.smallM}`, icon("ArrowTurnBackwardIcon", 12) + "Sent from Otto");
function delegationHeader(width = 806) {
  return div(`display:flex;flex-direction:column;gap:12px;width:${width}px;padding:14px 16px;border-radius:16px;border:1px solid ${C.z200};background-color:#FEFEFE;box-sizing:border-box`,
    div("display:flex;align-items:center;justify-content:space-between;gap:12px", row(10, avatarFor("Otto", 24) + icon("ArrowRight01Icon", 14, C.z400) + avatarFor("Alex", 24) + col(0, text(F.bodyM, "Delegated by Otto · 10:41") + muted(F.small, "Part of “Onboarding revamp PRD + estimate” · you own the outcome"))) + row(8, badge("accent", "Working for Otto") + button("secondary", "Back to Otto's thread", { size: "xs", leadingIcon: "ArrowLeft01Icon" }))) +
    div(`display:flex;gap:24px`, col(4, text(F.eyebrow, "Brief") + text(F.body, BRIEF), "flex:1") + col(6, text(F.eyebrow, "Expected back") + row(8, icon("Flag01Icon", 16, C.z500) + text(F.body, "PRD draft as a file")) + row(8, icon("Calendar03Icon", 16, C.z500) + text(F.body, "By today 6:00 pm")) + row(8, icon("Wallet01Icon", 16, C.z500) + text(F.body, "$2.00 cap · $0.12 used")), "width:240px;flex-shrink:0")));
}
emit("expert-thread-header-1440", "delegate/expert-thread/header/1440", page1440(div(`display:flex;flex-direction:row;height:900px;position:relative`, div(`display:flex;flex-direction:column;flex:1;min-width:0;align-items:center;position:relative`,
  div(`display:flex;flex-direction:column;gap:24px;width:806px;padding-top:56px;flex:1`, delegationHeader() +
    div("display:flex;flex-direction:column;align-items:flex-end;gap:6px;width:806px", sentFrom + div(`max-width:600px;padding:12px 16px;border-radius:8px;background-color:#F5F5F5;${F.body};color:#0A0A0A`, esc("Delegated task from Otto (not the user): draft the PRD for the onboarding revamp. Goals, scope, acceptance criteria. Sources: Notion research notes, last week's funnel numbers."))) +
    col(4, toolChain([{ icon: "Notebook01Icon", text: 'Read "Onboarding research" in Notion', state: "done" }, { icon: "Notebook01Icon", text: 'Read "Funnel Q3" in Notion', state: "done" }, { icon: "File01Icon", text: "Reading the activation sheet", state: "running" }]) + assistantText(["On it. I'll label every claim FACT, INFERENCE or UNKNOWN and flag anything that needs the owner's call.", div(`display:flex;flex-direction:column;gap:6px;padding:10px 12px;border-radius:12px;background-color:${C.z50};border:1px solid ${C.z200};width:520px`, text(F.eyebrow, "Plan") + row(8, icon("Tick02Icon", 14, "#10B981") + text(F.body, "Read research and funnel sources")) + row(8, icon("Loading03Icon", 14, "#A855F7") + text(F.body, "Write goals and scope")) + row(8, icon("CircleIcon", 14, C.z400) + text(F.body, "Acceptance criteria and open questions")) + row(8, icon("CircleIcon", 14, C.z400) + text(F.body, "Save the file and report back to Otto")))]))) +
  div(`display:flex;flex-direction:column;width:806px;padding-bottom:24px`, dockBar({ icon: "spinner", title: "Write goals and scope", count: "2/4", width: 806 }) + composer(806, "Message Alex…", "Alex"))) + div(`position:absolute;right:16px;top:12px`, iconButton("Folder01Icon", { size: 32, extra: "border:none;box-shadow:none;background-color:transparent" }))) + fab()), 1440, 900);

// ---------- Home ----------
function sectionCard(title, ic, right, body, extra = "") { return card(div("display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid " + C.z200, row(8, icon(ic, 16, C.z600) + text(F.bodyM, title)) + right) + body, extra); }
const needsRow = (who, kind, title, sub, cta) => div(`display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid ${C.z100}`, avatarFor(who, 32) + col(2, row(8, text(F.bodyM, title) + badge(kind === "question" ? "warning" : "accent", kind === "question" ? "Question" : "Approval", "small")) + sec(F.small, sub), "flex:1;min-width:0") + button(kind === "approval" ? "primary" : "secondary", cta, { size: "xs" }));
const workRow = (who, title, meta, status) => div(`display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid ${C.z100}`, row(2, avatarFor("Otto", 24) + (who ? icon("ArrowRight01Icon", 12, C.z400) + avatarFor(who, 24) : "")) + col(2, text(F.bodyM, title) + sec(F.small, meta), "flex:1;min-width:0") + status);
const teamRow = (who, status, sub) => div(`display:flex;align-items:center;gap:12px;padding:10px 16px;border-bottom:1px solid ${C.z100}`, avatarFor(who, 36) + col(2, row(6, text(F.bodyM, who) + status) + sec(F.small, sub), "flex:1;min-width:0") + row(4, iconButton("Comment01Icon", { size: 28 }) + iconButton("Settings01Icon", { size: 28 })));
emit("home-1440", "delegate/home/1440", page1440(div(`display:flex;flex-direction:column;align-items:center;height:900px;position:relative`, div(`display:flex;flex-direction:column;gap:24px;width:1116px;padding-top:32px`,
  div("display:flex;align-items:flex-start;justify-content:space-between", col(2, text(F.h4, "Good morning, Design") + sec(F.body, "2 things need you · 2 experts working for Otto")) + col(0, text(F.h5, "Sunday"), "align-items:flex-end")) +
  sectionCard("Needs you", "Notification01Icon", muted(F.small, "2 items"), col(0, needsRow("Alex", "question", "Q4 release train or December mini-launch?", "Alex, working for Otto on “Onboarding revamp PRD” · asked 3m ago", "Answer") + needsRow("Devon", "approval", "Devon wants to run the retention policy check", "Otto → Devon · $0.04 so far · ask-first mode", "Approve"))) +
  div("display:flex;gap:24px;align-items:flex-start",
    sectionCard("Recent work", "Task01Icon", muted(F.small, "This week · 4 completed"), col(0, workRow("Alex", "Onboarding revamp PRD", "Delegated 10:41 · draft returned 10:48 · 1 file", stBadge("done")) + workRow("Devon", "Eng estimate for onboarding revamp", "Delegated 10:49 · working 1m · $0.04", stBadge("working")) + workRow(null, "Weekly competitor watch", "Otto's routine · ran 08:00 · 3 changes found", stBadge("done")) + workRow("Anika", "Partner outreach list", "Delegated yesterday · budget cap hit", stBadge("failed"))), "flex:1") +
    col(24, sectionCard("Your team", "UserGroupIcon", muted(F.small, "2 working · 1 ready"), col(0, teamRow("Alex", stBadge("working"), "PRD for Otto · 2m · $0.12") + teamRow("Devon", stBadge("working"), "Eng estimate for Otto · 1m") + teamRow("Anika", badge("success", "Ready"), "Budget paused yesterday · $0 / $5"))) +
      sectionCard("Now & next", "Calendar03Icon", "", div("padding:12px 16px", col(6, text(F.eyebrow, "Running") + row(8, icon("Loading03Icon", 16, C.accent) + text(F.body, "Alex · PRD draft · 2m")) + row(8, icon("Loading03Icon", 16, C.accent) + text(F.body, "Devon · eng estimate · 1m")) + text(F.eyebrow, "Coming up") + row(8, icon("Clock01Icon", 16, C.z500) + text(F.body, "Otto · daily briefing · tomorrow 8:00")) + row(8, icon("Clock01Icon", 16, C.z500) + text(F.body, "Alex · weekly product review · Mon 9:00")))), "width:360px;flex-shrink:0"))))) + fab(), { active: "home" }), 1440, 900);

// ---------- Expert detail: Work tab (delegations come only from Otto) ----------
function tabs(items, active) { return row(0, items.map((t) => div(`display:flex;align-items:center;gap:6px;padding:0 4px 10px 4px;margin-right:20px;white-space:nowrap;flex-shrink:0;border-bottom:2px solid ${t === active ? C.black : "transparent"};${t === active ? F.bodyM : F.body};color:${t === active ? C.black : C.z600}`, esc(t))).join(""), `border-bottom:1px solid ${C.z200};width:100%`); }
const listRow = (who, title, desc, hint, status, time) => div(`display:flex;align-items:center;gap:16px;padding:14px 4px;border-bottom:1px solid ${C.z100}`, avatarFor(who, 36) + col(2, text(F.bodyM, title) + sec(F.body, desc) + (hint ? muted(F.small, hint) : ""), "flex:1;min-width:0") + div(`display:flex;align-items:center;gap:16px;flex-shrink:0`, div(`${F.small};color:${C.z500};width:72px;text-align:right`, esc(time)) + div("display:flex;justify-content:flex-end;width:112px", status) + icon("ArrowRight01Icon", 16, C.z400)));
const listFilters = (items, on) => row(4, items.map((l, i) => div(`height:32px;padding:0 12px;border-radius:9999px;display:flex;align-items:center;white-space:nowrap;flex-shrink:0;${F.body};${i === on ? `background-color:${C.z100};color:${C.black};font-weight:500` : `color:${C.z600}`}`, esc(l))).join(""));
const listHeader = (summary, right, filters, on) => col(16, div("display:flex;align-items:center;justify-content:space-between;gap:16px", text(F.body, summary) + right) + listFilters(filters, on), "padding:8px 0 4px 0");
emit("expert-detail-work-1440", "delegate/expert-detail/work-tab/1440", page1440(div(`display:flex;flex-direction:column;align-items:center;height:900px;position:relative`, div(`display:flex;flex-direction:column;gap:20px;width:1116px;padding-top:24px`,
  row(6, icon("ArrowLeft01Icon", 14, C.z500) + sec(F.body, "Back to Team")) +
  div("display:flex;align-items:center;justify-content:space-between", row(14, avatarFor("Alex", 56) + col(2, row(6, text(F.h4, "Alex") + sec(F.body, "· Product Manager")) + row(6, chip("Development", "ZapIcon") + badge("accent", "Working for Otto")))) + row(8, button("secondary", "Edit Soul", { size: "small", leadingIcon: "Edit02Icon" }) + button("primary", "Chat", { size: "small", leadingIcon: "Comment01Icon" }))) +
  tabs(["Basics", "Work", "Schedules", "Workflows", "Computer", "Integrations", "Skills", "Settings"], "Work") +
  div("display:flex;flex-direction:column;width:1116px;gap:4px",
    listHeader("5 delegations this week · 4 done · 1 failed · $1.67 spent", muted(F.small, "All from Otto"), ["All", "In progress", "Needs review", "Completed", "Failed"], 0) +
    listRow("Alex", "Onboarding revamp PRD", "PRD draft with 9 acceptance criteria and 3 open questions.", "From Otto · today 10:41 · 6m 40s · $0.31 · 1 file", stBadge("done"), "10:48") +
    listRow("Alex", "Competitor pricing read", "Pricing table for 6 competitors with sources.", "From Otto · yesterday · 9m · $0.52", stBadge("done"), "Yesterday") +
    listRow("Alex", "Roadmap re-prioritisation", "Stopped before the scoring step.", "From Otto · Thu · budget cap reached", stBadge("failed"), "Thu") +
    listRow("Alex", "Interview synthesis", "Themes from 8 interviews, quotes labelled by confidence.", "From Otto · Wed · 12m · $0.66 · 2 files", stBadge("done"), "Wed") +
    listRow("Alex", "Launch checklist", "Checklist for the December mini-launch.", "From Otto · Tue · 4m · $0.18", stBadge("done"), "Tue")))) + fab(), { active: "team" }), 1440, 900);

// ---------- Otto page: Delegations + Settings ----------
function ottoPage(activeTab, body) {
  return page1440(div(`display:flex;flex-direction:column;align-items:center;height:900px;position:relative`, div(`display:flex;flex-direction:column;gap:20px;width:1116px;padding-top:24px`,
    row(6, icon("ArrowLeft01Icon", 14, C.z500) + sec(F.body, "Back to Team")) +
    div("display:flex;align-items:center;justify-content:space-between", row(14, avatarFor("Otto", 56) + col(2, row(6, text(F.h4, "Otto") + sec(F.body, "· Head of AI")) + row(6, chip("Coordinates the team", "UserGroupIcon") + badge("accent", "3 delegations today")))) + row(8, button("primary", "Chat", { size: "small", leadingIcon: "Comment01Icon" }))) +
    tabs(["Basics", "Delegations", "Schedules", "Workflows", "Skills", "Settings"], activeTab) + body)) + fab(), { active: "team" });
}
emit("autopilot-delegations-1440", "delegate/autopilot/delegations/1440", ottoPage("Delegations", div("display:flex;flex-direction:column;width:1116px;gap:4px",
  listHeader("3 delegations today · 2 working · 1 needs you · $0.40 spent", row(6, icon("ShieldKeyIcon", 14, C.z500) + muted(F.small, "Ask first") + div(`${F.small};color:${C.z600};text-decoration-line:underline`, "Change")), ["All", "Needs you", "Working", "Completed", "Failed"], 0) +
  listRow("Devon", "Retention policy check", "Devon wants to run the check before estimating.", "Otto → Devon · waiting for your approval", stBadge("proposed"), "10:51") +
  listRow("Devon", "Eng estimate for onboarding revamp", "Estimating from the acceptance criteria.", "Otto → Devon · working · $0.04", stBadge("working"), "10:49") +
  listRow("Alex", "Onboarding revamp PRD", "PRD draft with 9 acceptance criteria and 3 open questions.", "Otto → Alex · 6m 40s · $0.31 · 1 file", stBadge("done"), "10:48") +
  listRow("Anika", "Partner outreach list", "Stopped at the weekly budget cap.", "Otto → Anika · yesterday", stBadge("failed"), "Yesterday"))), 1440, 900);

const selectCtl = (v) => div(`display:flex;align-items:center;justify-content:space-between;gap:8px;height:36px;width:160px;padding:0 10px 0 12px;border-radius:8px;border:1px solid ${C.z200};background-color:#FEFEFE;${F.body};box-sizing:border-box;flex-shrink:0`, esc(v) + icon("ArrowDown01Icon", 14, C.z500));
const toggleCtl = (on) => div(`width:36px;height:20px;border-radius:9999px;background-color:${on ? C.z800 : C.z300};display:flex;align-items:center;padding:2px;justify-content:${on ? "flex-end" : "flex-start"};box-sizing:border-box;flex-shrink:0`, div("width:16px;height:16px;border-radius:9999px;background-color:#FEFEFE"));
const settingRow = (title, desc, hint, control) => div(`display:flex;align-items:flex-start;justify-content:space-between;gap:24px;padding:20px 0;border-bottom:1px solid ${C.z200}`, col(4, text(F.bodyM, title) + sec(F.body, desc) + (hint ? muted(F.small, hint) : ""), "flex:1;min-width:0") + control);
emit("autopilot-settings-1440", "delegate/autopilot/settings/1440", ottoPage("Settings", div("display:flex;flex-direction:column;width:900px",
  settingRow("Delegation mode", "How Otto hands work to your experts.", "Ask first proposes each hand-off for approval · Auto lets a judge check each call · Unsupervised runs within caps", selectCtl("Ask first")) +
  settingRow("Per-delegation cap", "The most Otto can spend on a single hand-off.", "Weekly caps live on each expert", selectCtl("$2.00")) +
  settingRow("Daily delegation budget", "Otto stops delegating for the day once this is used up.", "", selectCtl("$10.00")) +
  settingRow("Ask before sending anything outside the workspace", "Email, Slack, posts and partner notes always wait for you, whatever the mode.", "", toggleCtl(true)) +
  settingRow("Ask before going over the cap", "Otto asks instead of stopping when a hand-off needs more than the per-delegation cap.", "", toggleCtl(true)) +
  settingRow("New experts start in ask-first", "Experts hired this week get approval on every hand-off until you switch them.", "", toggleCtl(false)))), 1440, 900);

// ---------- Mobile: Home and the expert Work tab at 390 ----------
const needsRowM = (who, kind, title, sub, cta) => div(`display:flex;gap:12px;padding:12px 16px;border-bottom:1px solid ${C.z100}`, avatarFor(who, 32) + col(6, badge(kind === "question" ? "warning" : "accent", kind === "question" ? "Question" : "Approval", "small") + text(F.bodyM, title) + sec(F.small, sub) + div("padding-top:2px", button(kind === "approval" ? "primary" : "secondary", cta, { size: "xs" })), "flex:1;min-width:0;align-items:flex-start"));
const workRowM = (who, title, meta, status) => div(`display:flex;align-items:center;gap:10px;padding:12px 16px;border-bottom:1px solid ${C.z100}`, row(2, avatarFor("Otto", 24) + (who ? icon("ArrowRight01Icon", 12, C.z400) + avatarFor(who, 24) : "")) + col(2, text(F.bodyM, title) + sec(F.small, meta), "flex:1;min-width:0") + status);
const teamRowM = (who, status, sub) => div(`display:flex;align-items:center;gap:12px;padding:10px 16px;border-bottom:1px solid ${C.z100}`, avatarFor(who, 36) + col(2, row(6, text(F.bodyM, who) + status) + sec(F.small, sub), "flex:1;min-width:0") + iconButton("Comment01Icon", { size: 28 }));
emit("home-390", "delegate/home/390", page390(div(`display:flex;flex-direction:column;align-items:center;flex:1`, div(`display:flex;flex-direction:column;gap:16px;width:342px;padding:76px 0 24px 0`,
  col(2, text(F.h4, "Good morning, Design") + sec(F.body, "2 things need you · 2 experts working for Otto")) +
  sectionCard("Needs you", "Notification01Icon", muted(F.small, "2 items"), col(0, needsRowM("Alex", "question", "Q4 release train or December mini-launch?", "Alex, for Otto on “Onboarding revamp PRD” · 3m ago", "Answer") + needsRowM("Devon", "approval", "Devon wants to run the retention policy check", "Otto → Devon · $0.04 so far · ask-first", "Approve"))) +
  sectionCard("Recent work", "Task01Icon", muted(F.small, "4 this week"), col(0, workRowM("Alex", "Onboarding revamp PRD", "10:41 → 10:48 · 1 file", stBadge("done")) + workRowM("Devon", "Eng estimate for onboarding", "10:49 · working 1m · $0.04", stBadge("working")) + workRowM(null, "Weekly competitor watch", "Otto's routine · 08:00 · 3 changes", stBadge("done")) + workRowM("Anika", "Partner outreach list", "Yesterday · budget cap hit", stBadge("failed")))) +
  sectionCard("Your team", "UserGroupIcon", muted(F.small, "2 working · 1 ready"), col(0, teamRowM("Alex", stBadge("working"), "PRD for Otto · 2m · $0.12") + teamRowM("Devon", stBadge("working"), "Eng estimate for Otto · 1m") + teamRowM("Anika", badge("success", "Ready"), "Budget paused yesterday · $0 / $5"))) +
  sectionCard("Now & next", "Calendar03Icon", "", div("padding:12px 16px", col(6, text(F.eyebrow, "Running") + row(8, icon("Loading03Icon", 16, C.accent) + text(F.body, "Alex · PRD draft · 2m")) + row(8, icon("Loading03Icon", 16, C.accent) + text(F.body, "Devon · eng estimate · 1m")) + text(F.eyebrow, "Coming up") + row(8, icon("Clock01Icon", 16, C.z500) + text(F.body, "Otto · daily briefing · tomorrow 8:00")) + row(8, icon("Clock01Icon", 16, C.z500) + text(F.body, "Alex · weekly product review · Mon 9:00"))))))), { height: 1420 }), 390, 1420);

const listRowM = (who, title, desc, hint, status) => div(`display:flex;gap:12px;padding:14px 0;border-bottom:1px solid ${C.z100}`, avatarFor(who, 36) + col(4, div("display:flex;align-items:flex-start;justify-content:space-between;gap:8px", text(F.bodyM, title) + status) + sec(F.body, desc) + (hint ? muted(F.small, hint) : ""), "flex:1;min-width:0"));
emit("expert-detail-work-390", "delegate/expert-detail/work-tab/390", page390(div(`display:flex;flex-direction:column;align-items:center;flex:1`, div(`display:flex;flex-direction:column;gap:16px;width:342px;padding:76px 0 24px 0`,
  row(6, icon("ArrowLeft01Icon", 14, C.z500) + sec(F.body, "Back to Team")) +
  row(14, avatarFor("Alex", 56) + col(2, text(F.h4, "Alex") + sec(F.body, "Product Manager"))) +
  row(8, chip("Development", "ZapIcon") + badge("accent", "Working for Otto")) +
  row(8, button("secondary", "Edit Soul", { size: "small", leadingIcon: "Edit02Icon", extra: "flex:1" }) + button("primary", "Chat", { size: "small", leadingIcon: "Comment01Icon", extra: "flex:1" })) +
  div("width:342px;overflow:hidden", tabs(["Basics", "Work", "Schedules", "Workflows", "Computer", "Integrations", "Skills", "Settings"], "Work")) +
  col(4, text(F.body, "5 delegations this week · 4 done · 1 failed · $1.67 spent") + muted(F.small, "All from Otto")) +
  div("width:342px;overflow:hidden", listFilters(["All", "In progress", "Needs review", "Completed", "Failed"], 0)) +
  div("display:flex;flex-direction:column",
    listRowM("Alex", "Onboarding revamp PRD", "PRD draft with 9 acceptance criteria and 3 open questions.", "6m 40s · $0.31 · 1 file · 10:48", stBadge("done")) +
    listRowM("Alex", "Competitor pricing read", "Pricing table for 6 competitors with sources.", "9m · $0.52 · Yesterday", stBadge("done")) +
    listRowM("Alex", "Roadmap re-prioritisation", "Stopped before the scoring step.", "budget cap reached · Thu", stBadge("failed")) +
    listRowM("Alex", "Interview synthesis", "Themes from 8 interviews, quotes labelled by confidence.", "12m · $0.66 · 2 files · Wed", stBadge("done")) +
    listRowM("Alex", "Launch checklist", "Checklist for the December mini-launch.", "4m · $0.18 · Tue", stBadge("done"))))), { height: 1060 }), 390, 1060);

// ---------- Flow storyboard ----------
const step = (n, title, trigger, body) => div(`display:flex;flex-direction:column;gap:12px;width:880px;padding:20px;border-radius:16px;background-color:${C.inset};border:1px solid ${C.z200}`, div("display:flex;align-items:center;justify-content:space-between", row(10, div(`display:flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:9999px;background-color:${C.z800};color:#FEFEFE;${F.smallM}`, String(n)) + text(F.h5, title)) + chip(trigger, "ArrowRight01Icon")) + body);
const FLOW = [
  [1, "User asks Otto", "types and sends", userMessage(ASK, 840)],
  [2, "Chain starts, approval node loads", "Otto calls delegate_to_expert", ottoTurn("skeleton", { width: 840 })],
  [3, "Approval sits in the chain (ask-first)", "held for approval", ottoTurn("proposed", { width: 840 })],
  [4, "Approved: row added, status line below", "click Approve", col(0, ottoTurn("approved", { width: 840 }) + dockFor("approved", 840))],
  [5, "Chain closed; loader line and dock open the panel", "sub-session running", col(0, ottoTurn("working", { width: 840 }) + dockFor("working", 840))],
  [6, "Question node in the chain", "elicitation", col(0, ottoTurn("needs-input", { width: 840 }) + dockFor("needs-input", 840))],
  [7, "Answered: chain closes again, status line returns", "user answers", ottoTurn("answered", { width: 840 })],
  [8, "Result returns", "sub-session completed", ottoTurn("done", { width: 840 })],
  [9, "Failure: status line with Retry", "budget cap hit", ottoTurn("failed", { width: 840 })],
];
emit("flow-storyboard", "delegate/flow/storyboard", div(`display:flex;flex-direction:row;gap:48px;padding:32px;align-items:flex-start;background-color:#FEFEFE`, FLOW.map(([n, t, tr, b]) => step(n, t, tr, b)).join("")), 9 * 880 + 8 * 48 + 64, 1150);

writeFileSync(join(out, "manifest.json"), JSON.stringify(entries, null, 2));
const NEW = ["copilot-needs-input-panel-answer-1440", "copilot-needs-input-panel-answered-1440", "copilot-multi-1440", "home-390", "expert-detail-work-390", "component-right-panel-detail"];
writeFileSync(join(out, "manifest-v3.json"), JSON.stringify(entries.filter((e) => NEW.includes(e.id)), null, 2));
console.log("entries", entries.length);
