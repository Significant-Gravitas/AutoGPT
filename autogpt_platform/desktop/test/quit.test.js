"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");

const { identity } = require("../src/identity");
const { applicationMenuTemplate } = require("../src/owner");
const {
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
} = require("../src/quit");
const { withUpdatesMenu } = require("../src/updater");

const normal = identity("");
const variant = identity("lab");
const oneRun = { graphRuns: 1, copilotTurns: 0 };
const busy = { graphRuns: 2, copilotTurns: 1 };

test("the runtime's count is read, and nothing else is taken for one", () => {
  assert.deepEqual(activityOf({ event: "activity", graph_runs: 2, copilot_turns: 1 }), busy);
  assert.deepEqual(activityOf({ event: "activity", graph_runs: 0, copilot_turns: 0 }), IDLE);
  assert.equal(activityOf({ event: "ready", url: "http://127.0.0.1:18473" }), null);
  assert.equal(activityOf({ event: "activity", graph_runs: "2", copilot_turns: 0 }), null);
  assert.equal(activityOf({ event: "activity", graph_runs: -1, copilot_turns: 0 }), null);
  assert.equal(activityOf({ event: "activity", graph_runs: 1 }), null);
  assert.equal(activityOf(null), null);
});

test("anything running counts, of either kind", () => {
  assert.equal(isActive(IDLE), false);
  assert.equal(isActive(oneRun), true);
  assert.equal(isActive({ graphRuns: 0, copilotTurns: 1 }), true);
});

test("what is running is said in words", () => {
  assert.equal(describeActivity(IDLE), null);
  assert.equal(describeActivity(oneRun), "1 agent run");
  // Not "turns": the app never calls them that where its user can read it.
  assert.equal(describeActivity({ graphRuns: 0, copilotTurns: 2 }), "2 AutoPilot replies");
  assert.equal(describeActivity(busy), "2 agent runs and 1 AutoPilot reply");
});

test("the tray menu has one line about it, and none when nothing runs", () => {
  assert.deepEqual(activityMenuItems(IDLE), []);
  assert.deepEqual(activityMenuItems(busy), [
    { label: "2 agent runs and 1 AutoPilot reply in progress", enabled: false },
  ]);
});

test("the tray icon says which install it is, what runs, and that it is being stopped", () => {
  assert.equal(trayToolTip(normal, IDLE), "AutoGPT");
  assert.equal(trayToolTip(variant, oneRun), "AutoGPT (lab): 1 agent run in progress");
  assert.equal(trayToolTip(variant, oneRun, true), "AutoGPT (lab): stopping 1 agent run");
  assert.equal(trayToolTip(variant, IDLE, true), "AutoGPT (lab)");
});

test("with nothing running a quit stops the runtime without asking", () => {
  assert.equal(quitStep({ activity: IDLE, quitting: false, sessionEnding: false, runtimeExited: false }), "stop");
});

test("with something running a quit asks", () => {
  assert.equal(quitStep({ activity: oneRun, quitting: false, sessionEnding: false, runtimeExited: false }), "ask");
  assert.equal(
    quitStep({ activity: { graphRuns: 0, copilotTurns: 1 }, quitting: false, sessionEnding: false, runtimeExited: false }),
    "ask",
  );
});

test("a log-off or a shutdown asks nothing, whatever is running", () => {
  assert.equal(quitStep({ activity: busy, quitting: false, sessionEnding: true, runtimeExited: false }), "stop");
});

test("a second quit while the runtime is stopping hurries it and still waits for it", () => {
  assert.equal(quitStep({ activity: busy, quitting: true, sessionEnding: false, runtimeExited: false }), "hurry");
  assert.equal(quitStep({ activity: IDLE, quitting: true, sessionEnding: false, runtimeExited: false }), "hurry");
});

test("with no runtime left to stop, a quit goes ahead", () => {
  // As after a crash, and as the updater's own quit once the runtime was stopped for it.
  assert.equal(quitStep({ activity: busy, quitting: true, sessionEnding: false, runtimeExited: true }), "leave");
  assert.equal(quitStep({ activity: busy, quitting: false, sessionEnding: false, runtimeExited: true }), "leave");
});

test("closing the window is a quit: it asks while something runs", () => {
  const open = { quitting: false, sessionEnding: false };
  assert.equal(closeStep({ ...open, activity: IDLE }), "close");
  assert.equal(closeStep({ ...open, activity: oneRun }), "ask");
});

test("a window brought back after the background was chosen asks again: nothing hides without a word", () => {
  // Hidden windows cannot be closed, so the one being closed was shown
  // again. The earlier answer does not stand in for this one: there is no
  // way to say it was given, and the shell keeps no such thing.
  assert.doesNotMatch(fs.readFileSync(path.join(__dirname, "..", "src", "main.js"), "utf8"), /\bbackground = /);
  assert.equal(closeStep({ activity: oneRun, quitting: false, sessionEnding: false, background: true }), "ask");
});

test("a window closes without a question when the app is quitting or the session is ending", () => {
  assert.equal(closeStep({ activity: busy, quitting: true, sessionEnding: false }), "close");
  assert.equal(closeStep({ activity: busy, quitting: false, sessionEnding: true }), "close");
});

const EITHER = [false, true];

test("every quit: what is done, whatever runs and whatever is already under way", () => {
  for (const activity of [IDLE, oneRun, busy]) {
    for (const quitting of EITHER) {
      for (const sessionEnding of EITHER) {
        for (const runtimeExited of EITHER) {
          const state = { activity, quitting, sessionEnding, runtimeExited };
          let expected = isActive(activity) && !sessionEnding ? "ask" : "stop";
          if (quitting) expected = "hurry";
          if (runtimeExited) expected = "leave";
          assert.equal(quitStep(state), expected, JSON.stringify(state));
        }
      }
    }
  }
});

test("every close of the window: it asks only when a user's own close would end something", () => {
  for (const activity of [IDLE, oneRun, busy]) {
    for (const quitting of EITHER) {
      for (const sessionEnding of EITHER) {
        const state = { activity, quitting, sessionEnding };
        const expected = isActive(activity) && !quitting && !sessionEnding ? "ask" : "close";
        assert.equal(closeStep(state), expected, JSON.stringify(state));
      }
    }
  }
});

test("the session ending stops the runtime, and hurries a stop that is waiting for runs to end", () => {
  assert.equal(sessionEndStep({ quitting: false, runtimeExited: false }), "stop");
  // The user chose "Stop and quit" and then logged off: the wait is over.
  assert.equal(sessionEndStep({ quitting: true, runtimeExited: false }), "hurry");
  assert.equal(sessionEndStep({ quitting: false, runtimeExited: true }), "nothing");
  assert.equal(sessionEndStep({ quitting: true, runtimeExited: true }), "nothing");
});

test("the system's word that the session is ending holds for a while, not for ever", () => {
  assert.equal(sessionIsEnding(null, 5_000_000), false);
  assert.equal(sessionIsEnding(1_000_000, 1_000_000), true);
  assert.equal(sessionIsEnding(1_000_000, 1_000_000 + SESSION_END_MS - 1), true);
  // A log-off that was called off: an hour later a quit is the user's own.
  assert.equal(sessionIsEnding(1_000_000, 1_000_000 + SESSION_END_MS), false);
  assert.equal(sessionIsEnding(1_000_000, 1_000_000 + 3_600_000), false);
  assert.ok(SESSION_END_MS >= 60_000 && SESSION_END_MS <= 300_000);
  const later = sessionIsEnding(1_000_000, 1_000_000 + 3_600_000);
  assert.equal(quitStep({ activity: busy, quitting: false, sessionEnding: later, runtimeExited: false }), "ask");
  assert.equal(closeStep({ activity: busy, quitting: false, sessionEnding: later }), "ask");
});

test("every answer: it does what its button says unless the question was overtaken", () => {
  const does = { [KEEP_RUNNING]: "background", [STOP_AND_QUIT]: "stop", [CANCEL]: "nothing" };
  for (const answer of [KEEP_RUNNING, STOP_AND_QUIT, CANCEL]) {
    for (const quitting of EITHER) {
      for (const runtimeExited of EITHER) {
        const state = { answer, quitting, runtimeExited };
        const expected = quitting || runtimeExited ? "nothing" : does[answer];
        assert.equal(answerStep(state), expected, JSON.stringify(state));
      }
    }
  }
  assert.equal(answerStep({ answer: undefined, quitting: false, runtimeExited: false }), "nothing");
});

test("a yes to the update counts only for the version asked about, still ready, with nothing else under way", () => {
  const yes = { response: 0, asked: "1.2.3", phase: "ready", version: "1.2.3", quitting: false, restarting: false };
  assert.equal(updateAnswer(yes), true);
  assert.equal(updateAnswer({ ...yes, response: 1 }), false);
  assert.equal(updateAnswer({ ...yes, response: undefined }), false);
  assert.equal(updateAnswer({ ...yes, version: "1.2.4" }), false);
  assert.equal(updateAnswer({ ...yes, phase: "downloading" }), false);
  assert.equal(updateAnswer({ ...yes, quitting: true }), false);
  assert.equal(updateAnswer({ ...yes, restarting: true }), false);
});

test("outside macOS the application menu has Quit, so the app can be quit without a tray icon", () => {
  for (const platform of ["linux", "win32"]) {
    const menus = withQuitMenu(applicationMenuTemplate(platform, []), variant, platform);
    const items = menus.flatMap((menu) => menu.submenu || []);
    const quit = items.filter((item) => item.role === "quit");
    assert.equal(quit.length, 1, platform);
    // The role is Electron's app.quit(), which goes through `before-quit` and its question.
    assert.equal(quit[0].label, "&Quit AutoGPT (lab)");
    assert.equal(quit[0].click, undefined);
    assert.equal(quit[0].accelerator, platform === "linux" ? "Ctrl+Q" : undefined);
    assert.equal(menus[0].label, "&File");
  }
  // macOS has it in the application menu already (owner.js, role appMenu).
  const mac = applicationMenuTemplate("darwin", []);
  assert.equal(withQuitMenu(mac, normal, "darwin"), mac);
  assert.ok(mac.some((menu) => menu.role === "appMenu"));
  // With the Updates menu, as the shell builds it.
  const built = withQuitMenu(withUpdatesMenu(applicationMenuTemplate("linux", []), [{ label: "x" }], "linux"), normal, "linux");
  assert.deepEqual(built.map((menu) => menu.label), ["&File", "&Account", "&Updates"]);
});

test("the question says what runs and offers the three ways out", () => {
  const question = quitQuestion({ install: normal, activity: busy, platform: "win32" });
  assert.equal(question.message, "2 agent runs and 1 AutoPilot reply are in progress.");
  assert.deepEqual(question.buttons, ["Keep running in the background", "Stop and quit AutoGPT", "Cancel"]);
  assert.match(question.detail, /notification area/);
  assert.match(question.detail, /do not start again/);
  // What "keep running" leaves behind, and how to end it: the app does not
  // close by itself when the runs are over.
  assert.match(question.detail, /^Keep running: AutoGPT stays open without a window/);
  assert.match(question.detail, /does not close by itself when they finish/);
  assert.match(question.detail, /To quit it later, use its icon in the notification area, or start AutoGPT again/);
  assert.match(question.detail, /\n\nStop and quit: they end now/);
  assert.doesNotMatch(question.detail, /until they have finished/);
  assert.doesNotMatch([question.message, question.detail].join(" "), /\bturns?\b/);
  // Enter keeps the runs, Escape cancels: neither ends anything.
  assert.equal(quitAnswer(question.defaultId), KEEP_RUNNING);
  assert.equal(quitAnswer(question.cancelId), CANCEL);
  assert.equal(quitQuestion({ install: normal, activity: oneRun, platform: "darwin" }).message, "1 agent run is in progress.");
  assert.match(quitQuestion({ install: normal, activity: oneRun, platform: "darwin" }).detail, /menu bar/);
  assert.match(quitQuestion({ install: normal, activity: oneRun, platform: "linux" }).detail, /system tray/);
});

test("the answers are the buttons, in order, and anything else is a cancel", () => {
  assert.deepEqual([0, 1, 2].map(quitAnswer), [KEEP_RUNNING, STOP_AND_QUIT, CANCEL]);
  assert.equal(quitAnswer(-1), CANCEL);
  assert.equal(quitAnswer(undefined), CANCEL);
  assert.equal(quitAnswer(7), CANCEL);
});

test("a variant's question and tray name the variant, never the normal app", () => {
  const question = quitQuestion({ install: variant, activity: oneRun, platform: "win32" });
  const said = [question.title, question.detail, ...question.buttons].join("\n");
  assert.equal(question.title, "AutoGPT (lab)");
  assert.deepEqual(question.buttons[1], "Stop and quit AutoGPT (lab)");
  assert.ok(!/AutoGPT(?! \(lab\))/.test(said), said);
  const update = updateQuestion({ install: variant, version: "1.2.3", activity: oneRun });
  assert.ok(!/AutoGPT(?! \(lab\))/.test([update.title, update.message, update.detail].join("\n")));
});

test("restarting to update asks once: with something running it says what a yes stops", () => {
  const idle = updateQuestion({ install: normal, version: "1.2.3", activity: IDLE });
  assert.equal(idle.message, "Restart AutoGPT to update to 1.2.3?");
  assert.deepEqual(idle.buttons, ["Restart", "Later"]);
  assert.equal(idle.detail, undefined);
  assert.equal(idle.cancelId, 1);

  const running = updateQuestion({ install: normal, version: "1.2.3", activity: busy });
  assert.deepEqual(running.buttons, ["Stop and restart", "Later"]);
  assert.match(running.detail, /^2 agent runs and 1 AutoPilot reply are in progress\./);
  // Enter does not end anybody's runs.
  assert.equal(running.defaultId, 1);
  assert.equal(running.cancelId, 1);
});

// main.js is not loaded by any test (it needs Electron). These hold it to
// the decisions above by its text.
const MAIN = fs.readFileSync(path.join(__dirname, "..", "src", "main.js"), "utf8");

test("the shell leaves each decision about quitting to quit.js", () => {
  assert.match(MAIN, /mainWindow\.on\("close", \(event\) => \{[\s\S]*?const step = closeStep\(/);
  assert.match(MAIN, /const step = answerStep\(\{ answer, quitting, runtimeExited: runtimeExited\(\) \}\);/);
  assert.match(MAIN, /if \(sessionEndStep\(\{ quitting, runtimeExited: runtimeExited\(\) \}\) === "nothing"\) return;/);
  assert.match(MAIN, /updateAnswer\(\{ response, asked: version, \.\.\.updater\.state\(\), quitting, restarting \}\)/);
  assert.match(MAIN, /withQuitMenu\(withUpdates, install, process\.platform\)/);
  // Whether the session is ending is asked each time, never remembered as a yes.
  assert.doesNotMatch(MAIN, /sessionEnding = true/);
  assert.match(MAIN, /return sessionIsEnding\(sessionEndingSince, Date\.now\(\)\);/);
  assert.match(MAIN, /app\.on\("before-quit", [\s\S]*?const step = quitStep\(/);
  assert.match(MAIN, /quitAnswer\(await showQuestion\(quitQuestion\(\{ install, activity, platform: process\.platform \}\)\)\)/);
  assert.match(MAIN, /showQuestion\(updateQuestion\(\{ install, version, activity \}\)\)/);
  assert.match(MAIN, /\.\.\.activityMenuItems\(activity\),/);
  assert.match(MAIN, /trayToolTip\(install, activity, stoppingRuns\)/);
});

test("the shell keeps the count out of the start's history and forgets it with the runtime", () => {
  const hears = MAIN.slice(MAIN.indexOf("function onRuntimeEvent"), MAIN.indexOf("report(event);", MAIN.indexOf("function onRuntimeEvent")));
  assert.match(hears, /activityOf\(event\)/);
  const starts = MAIN.slice(MAIN.indexOf("function startRuntime"), MAIN.indexOf("function onRuntimeEvent"));
  assert.match(starts, /activity = IDLE;\s+claudeCode\.runtimeStarting\(\);/);
  // A runtime that crashed runs nothing: closing the window then must not ask.
  assert.match(starts, /started\.on\("exit", [^\n]+\n[^\n]+\n\s+if \(runtime === started\) showActivity\(IDLE\);/);
});

test("only the user's own choice stops the runs: every other stop is the plain one", () => {
  const stops = [...MAIN.matchAll(/\.stop\(([^)]*)\)/g)].map((match) => match[1]);
  assert.deepEqual(stops.filter((argument) => argument !== "").sort(), ["{ stopRuns }", "{ stopRuns }"]);
  assert.match(MAIN, /if \(step === "stop"\) quit\(\{ stopRuns: true \}\);/);
  // The session ending, and a signal: no question.
  assert.match(MAIN, /window\.on\("session-end", endWithTheSession\)/);
  assert.match(MAIN, /powerMonitor\.on\("shutdown", \(\) => \{\s+sessionEndingSince = Date\.now\(\);/);
  assert.match(MAIN, /for \(const signal of \["SIGTERM", "SIGHUP", "SIGINT"\]\)/);
});
