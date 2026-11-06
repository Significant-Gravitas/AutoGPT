"use strict";

// What a quit does while agents are running, without Electron so it can be
// tested.
//
// The runtime says how many agent runs and AutoPilot turns (to the user:
// AutoPilot replying) are in progress, whenever that changes
// (runtime/autogpt_desktop/runs.py):
//
//   {"event": "activity", "graph_runs": 2, "copilot_turns": 1}
//
// With none, quitting is immediate. With any, the user decides:
//
//   keep running   the windows hide and the app stays, with its tray icon.
//                  It quits only when it is asked to again, and a window
//                  that was brought back asks again when it is closed:
//                  nothing is ever hidden, or ended, without a word.
//   stop and quit  the runtime is asked to end what is in progress
//                  (runtime.js `stop({ stopRuns: true })`). Runs ended this
//                  way do not start again.
//   cancel         nothing happens.
//
// A quit nobody chose (the system shutting down, a log-off, a signal) asks
// nothing and stops the plain way: what was running resumes on the next
// start. Installing an update asks its own question once, which says what
// restarting will stop.

const IDLE = Object.freeze({ graphRuns: 0, copilotTurns: 0 });

const KEEP_RUNNING = "background";
const STOP_AND_QUIT = "stop";
const CANCEL = "cancel";
// The order of the buttons.
const ANSWERS = [KEEP_RUNNING, STOP_AND_QUIT, CANCEL];

// How long the system's word that the session is ending holds. A log-off
// can be called off (another app has unsaved work), and nobody says so: a
// quit long after it is the user's own again, and asks.
const SESSION_END_MS = 120_000;

// The counts an `activity` event carries, or null for any other event and
// for one this shell cannot read.
function activityOf(event) {
  if (!event || event.event !== "activity") return null;
  const { graph_runs: graphRuns, copilot_turns: copilotTurns } = event;
  if (![graphRuns, copilotTurns].every((count) => Number.isSafeInteger(count) && count >= 0)) return null;
  return { graphRuns, copilotTurns };
}

function isActive(activity) {
  return activity.graphRuns + activity.copilotTurns > 0;
}

// "2 agent runs and 1 AutoPilot reply", or null with nothing running.
function describeActivity({ graphRuns, copilotTurns }) {
  const parts = [];
  if (graphRuns) parts.push(`${graphRuns} agent ${graphRuns === 1 ? "run" : "runs"}`);
  if (copilotTurns) parts.push(`${copilotTurns} AutoPilot ${copilotTurns === 1 ? "reply" : "replies"}`);
  return parts.length ? parts.join(" and ") : null;
}

function inProgress(activity) {
  const what = describeActivity(activity);
  if (!what) return null;
  return `${what} ${activity.graphRuns + activity.copilotTurns === 1 ? "is" : "are"} in progress`;
}

// For the tray menu: one line, there only while something runs. Disabled:
// there is nothing to click.
function activityMenuItems(activity) {
  const what = describeActivity(activity);
  return what ? [{ label: `${what} in progress`, enabled: false }] : [];
}

// The tray icon's tooltip.
function trayToolTip(install, activity, stoppingRuns = false) {
  const what = describeActivity(activity);
  if (!what) return install.productName;
  return `${install.productName}: ${stoppingRuns ? `stopping ${what}` : `${what} in progress`}`;
}

function trayPlace(platform) {
  if (platform === "darwin") return "the menu bar";
  return platform === "win32" ? "the notification area" : "the system tray";
}

// Electron's dialog.showMessageBox options. `quitAnswer` reads the reply.
function quitQuestion({ install, activity, platform }) {
  const name = install.productName;
  return {
    type: "question",
    title: name,
    message: `${inProgress(activity)}.`,
    detail:
      `Keep running: ${name} stays open without a window and they carry on. It does not close ` +
      `by itself when they finish. To quit it later, use its icon in ${trayPlace(platform)}, ` +
      `or start ${name} again to bring the window back.\n\n` +
      "Stop and quit: they end now and do not start again.",
    buttons: ["Keep running in the background", `Stop and quit ${name}`, "Cancel"],
    defaultId: ANSWERS.indexOf(KEEP_RUNNING),
    cancelId: ANSWERS.indexOf(CANCEL),
    noLink: true,
  };
}

// Anything but the two buttons that do something (a closed dialog, a reply
// from a dialog that could not be shown) is a cancel.
function quitAnswer(response) {
  return ANSWERS[response] || CANCEL;
}

// What an answer to the question does: "background", "stop" (the runs, and
// quit) or "nothing". The question may have been overtaken while it was on
// screen, by another quit or by a log-off: then the answer counts for
// nothing.
function answerStep({ answer, quitting, runtimeExited }) {
  if (quitting || runtimeExited) return "nothing";
  if (answer === KEEP_RUNNING) return "background";
  return answer === STOP_AND_QUIT ? "stop" : "nothing";
}

// "Restart to update": the same decision in one question. Not restarting is
// both "keep running" and "cancel", so there are two buttons.
function updateQuestion({ install, version, activity }) {
  const running = inProgress(activity);
  return {
    type: "question",
    title: install.productName,
    message: `Restart ${install.productName} to update to ${version}?`,
    ...(running ? { detail: `${running}. Restarting ends them now, and they do not start again.` } : {}),
    buttons: [running ? "Stop and restart" : "Restart", "Later"],
    defaultId: running ? 1 : 0,
    cancelId: 1,
    noLink: true,
  };
}

// Whether the reply to `updateQuestion` is a yes that still counts: the
// first button, with nothing changed while the question was on screen
// (`asked` is the version it named, `phase` and `version` the updater's
// state now).
function updateAnswer({ response, asked, phase, version, quitting, restarting }) {
  return response === 0 && !quitting && !restarting && phase === "ready" && version === asked;
}

// Whether the system's word that the session is ending (`since`, a time in
// milliseconds, or null when it never said so) still holds at `now`.
function sessionIsEnding(since, now) {
  return since !== null && now - since < SESSION_END_MS;
}

// What `before-quit` does:
//   "leave"  let the quit go ahead: there is no runtime to wait for
//   "hurry"  a stop is already under way; a second quit makes it wait for
//            nothing more (and the app still exits only once it has ended)
//   "ask"    put the question
//   "stop"   stop the runtime the plain way, then exit
function quitStep({ activity, quitting, sessionEnding, runtimeExited }) {
  if (runtimeExited) return "leave";
  if (quitting) return "hurry";
  return isActive(activity) && !sessionEnding ? "ask" : "stop";
}

// What closing the main window does: "close" (which quits the app) or
// "ask". After "keep running" the window is hidden, not closed, so one that
// is being closed was brought back, and is asked about like any other.
function closeStep({ activity, quitting, sessionEnding }) {
  return quitting || sessionEnding || !isActive(activity) ? "close" : "ask";
}

// What the session ending does (Windows says so to every window, with no
// `before-quit`): "stop" the runtime the plain way, "hurry" a stop that is
// already under way, as a second quit does, or "nothing".
function sessionEndStep({ quitting, runtimeExited }) {
  if (runtimeExited) return "nothing";
  return quitting ? "hurry" : "stop";
}

// Adds Quit to owner.js's application menu where the system gives none:
// everywhere but macOS, whose application menu has it. The tray icon's Quit
// is not always there (GNOME shows no tray without an extension), and an
// app that was told to keep running must not be one that cannot be quit.
// Ctrl+Q on Linux, where that is what it means; Windows has Alt+F4, which
// closes the window and so asks the same question.
function withQuitMenu(template, install, platform) {
  if (platform === "darwin") return template;
  const quit = {
    label: `&Quit ${install.productName}`,
    role: "quit",
    ...(platform === "win32" ? {} : { accelerator: "Ctrl+Q" }),
  };
  return [{ label: "&File", submenu: [quit] }, ...template];
}

module.exports = {
  CANCEL,
  IDLE,
  KEEP_RUNNING,
  SESSION_END_MS,
  STOP_AND_QUIT,
  activityMenuItems,
  activityOf,
  answerStep,
  closeStep,
  describeActivity,
  isActive,
  quitAnswer,
  quitQuestion,
  quitStep,
  sessionEndStep,
  sessionIsEnding,
  trayToolTip,
  updateAnswer,
  updateQuestion,
  withQuitMenu,
};
