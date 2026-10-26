"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");

const { claudeCodeLine, claudeCodeMenuItems, claudeCodeReports, claudeCodeStatus } = require("../src/claude-code");
const { identity } = require("../src/identity");
const { applicationMenuTemplate } = require("../src/owner");
const { parseEvent } = require("../src/runtime");

const DESKTOP = path.join(__dirname, "..");
const NORMAL = identity("");
const VARIANT = identity("voice");

function reported(state) {
  const line = JSON.stringify({
    event: "claude_code",
    state,
    message: "for the log",
    cli: "C:\\Users\\someone\\.local\\bin\\claude.exe",
    version: "2.1.284",
    bundled: false,
  });
  return claudeCodeStatus(parseEvent(line));
}

test("with the sign-in in use, the line says whose plan pays and that a restart is needed", () => {
  const line = claudeCodeLine(reported("in_use"), NORMAL);
  assert.match(line, /^Self-hosted chat connection: Claude Code sign-in/);
  assert.match(line, /turns count against your Claude plan/);
  assert.match(line, /Restart AutoGPT after signing in or out/);
});

test("signed out or not installed, the one instruction is to run claude and sign in", () => {
  for (const state of ["signed_out", "not_found"]) {
    const line = claudeCodeLine(reported(state), NORMAL);
    assert.match(line, /^Self-hosted chat connection: configure API keys or a local model in the settings file/);
    assert.match(line, /run `claude`, sign in, and restart AutoGPT/);
  }
});

test("turned off or refused, the line says the sign-in is not used and where to look", () => {
  assert.match(claudeCodeLine(reported("off"), NORMAL), /^Self-hosted chat connection:.*sign-in is turned off\.$/);
  assert.match(claudeCodeLine(reported("refused"), NORMAL), /the log says why/);
});

test("a CLI that did not answer is not called signed out: the line asks for a restart, not a sign-in", () => {
  const line = claudeCodeLine(reported("no_answer"), NORMAL);
  assert.match(line, /^Self-hosted chat connection: configure API keys or a local model in the settings file/);
  assert.match(line, /did not say whether it is signed in; restart AutoGPT to ask it again/);
  assert.doesNotMatch(line, /sign in,/);
});

test("a variant names itself, not the normal app", () => {
  assert.match(claudeCodeLine(reported("in_use"), VARIANT), /Restart AutoGPT \(voice\) after/);
  assert.match(claudeCodeLine(reported("signed_out"), VARIANT), /restart AutoGPT \(voice\)\.$/);
});

test("every state the runtime reports has a line, and nothing else has one", () => {
  const runtime = fs.readFileSync(path.join(DESKTOP, "runtime", "autogpt_desktop", "claude_code.py"), "utf8");
  const states = [...runtime.matchAll(/^(?:IN_USE|SIGNED_OUT|NO_ANSWER|NOT_FOUND|OFF|REFUSED) = "(\w+)"$/gm)].map((m) => m[1]);
  assert.equal(states.length, 6, "claude_code.py no longer defines its six states where this test reads them");
  for (const state of states) assert.ok(claudeCodeLine(reported(state), NORMAL), state);
  assert.equal(reported("something_newer"), null);
  assert.equal(reported("constructor"), null);
  assert.equal(claudeCodeStatus(null), null);
  assert.equal(claudeCodeStatus({ event: "ready", state: "in_use" }), null);
});

test("the menus get one line that cannot be clicked, and none before the runtime has reported", () => {
  assert.deepEqual(claudeCodeMenuItems({ status: null, install: NORMAL }), []);
  const items = claudeCodeMenuItems({ status: reported("in_use"), install: NORMAL });
  assert.equal(items.length, 1);
  assert.equal(items[0].enabled, false);
  assert.equal(items[0].click, undefined);
  assert.equal(items[0].label, claudeCodeLine(reported("in_use"), NORMAL));
});

test("startup auth status describes the Self-hosted chat connection, not a chat's selected subscription", () => {
  for (const state of ["in_use", "signed_out", "no_answer", "not_found", "off", "refused"]) {
    const [{ label }] = claudeCodeMenuItems({ status: reported(state), install: NORMAL });
    assert.match(label, /^Self-hosted chat connection:/);
    assert.doesNotMatch(label, /AutoPilot uses|default connection|current connection/i);
  }
});

test("nothing the runtime found is shown: no path, no version, and never an account", () => {
  const status = reported("in_use");
  assert.deepEqual(status, { state: "in_use" });
  const withAccount = claudeCodeStatus({ event: "claude_code", state: "in_use", email: "someone@example.com" });
  assert.deepEqual(withAccount, { state: "in_use" });
  assert.doesNotMatch(claudeCodeLine(status, NORMAL), /claude\.exe|2\.1\.284|someone/);
});

test("the line is in the tray menu and, on macOS, in the application menu", () => {
  const main = fs.readFileSync(path.join(DESKTOP, "src", "main.js"), "utf8");
  assert.match(main, /applicationMenuTemplate\(process\.platform, \[\.\.\.ownerItems\(\), \.\.\.autopilotItems\(\)\]\)/);
  const tray = main.slice(main.indexOf("function trayMenu()"), main.indexOf("function ownerItems()"));
  assert.match(tray, /\.\.\.autopilotItems\(\),/);
  // Both menus are drawn again whenever the line changes, and the line is the
  // one main.js keeps in claudeCodeReports.
  assert.match(main, /const claudeCode = claudeCodeReports\(\(\) => refreshMenus\(\)\);/);
  assert.match(main, /claudeCodeMenuItems\(\{ status: claudeCode\.status\(\), install \}\)/);
  const heard = main.slice(main.indexOf("function onRuntimeEvent("), main.indexOf("function lookForAFixedVersion()"));
  assert.match(heard, /^function onRuntimeEvent\(event\) \{\r?\n  if \(claudeCode\.hears\(event\)\) return;/);
  // A runtime that is started again has not reported yet: said before it starts.
  const start = main.slice(main.indexOf("function startRuntime()"), main.indexOf("function onRuntimeEvent("));
  assert.ok(start.includes("claudeCode.runtimeStarting();"));
  assert.ok(start.indexOf("claudeCode.runtimeStarting();") < start.indexOf("started.start();"));

  const line = claudeCodeMenuItems({ status: reported("in_use"), install: NORMAL });
  const mac = applicationMenuTemplate("darwin", [{ label: "Reset owner password…" }, ...line]);
  const account = mac.find((menu) => menu.label === "Account");
  assert.equal(account.submenu.at(-1).label, line[0].label);
});

test("the line of the last runtime goes as soon as another is started, and the menus are redrawn", () => {
  let redrawn = 0;
  const reports = claudeCodeReports(() => redrawn++);
  assert.equal(reports.status(), null);

  assert.equal(reports.hears(parseEvent(JSON.stringify({ event: "claude_code", state: "in_use" }))), true);
  assert.deepEqual([reports.status(), redrawn], [{ state: "in_use" }, 1]);
  assert.equal(claudeCodeMenuItems({ status: reports.status(), install: NORMAL }).length, 1);

  // Signed out of Claude Code, then restarted: the new runtime has said
  // nothing yet, and says nothing at all if it fails before it looks.
  reports.runtimeStarting();
  assert.deepEqual([reports.status(), redrawn], [null, 2]);
  assert.deepEqual(claudeCodeMenuItems({ status: reports.status(), install: NORMAL }), []);

  assert.equal(reports.hears({ event: "progress", message: "Starting" }), false);
  assert.equal(reports.hears(null), false);
  assert.deepEqual([reports.status(), redrawn], [null, 2]);

  assert.equal(reports.hears({ event: "claude_code", state: "signed_out" }), true);
  assert.deepEqual([reports.status(), redrawn], [{ state: "signed_out" }, 3]);
  // A state this shell cannot word clears the line; it does not keep the old one.
  assert.equal(reports.hears({ event: "claude_code", state: "something_newer" }), true);
  assert.deepEqual([reports.status(), redrawn], [null, 4]);
});

test("the app has no Claude sign-in of its own: no window, field or token setting for one", () => {
  for (const name of fs.readdirSync(path.join(DESKTOP, "src"))) {
    if (!/\.(js|html)$/.test(name)) continue;
    const source = fs.readFileSync(path.join(DESKTOP, "src", name), "utf8");
    assert.doesNotMatch(source, /CLAUDE_CODE_OAUTH_TOKEN|claude\.ai\/oauth|claude login|setup-token/i, name);
  }
});
