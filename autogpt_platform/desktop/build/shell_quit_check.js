"use strict";

// Quit the real shell with agents running, and check what it does.
//
//   node build/shell_quit_check.js [--data-dir DIR] [--keep]
//
// Needs `npm ci` (for Electron) and an assembled runtime (build/runtime, or
// AUTOGPT_DESKTOP_RUNTIME), a display, and about 3 GB of free memory. It
// starts the shell from this tree on a data directory and a browser profile
// of its own (a temporary one, removed afterwards, unless --data-dir names
// one), so it touches neither an installed AutoGPT nor its data; the
// installed app may be running meanwhile.
//
// The shell is driven through two debugging ports: Chromium's, for the
// window (sign up, build an agent of one timer block, run it), and Node's
// (`--inspect`), for the main process. Through the second, the one thing
// that cannot be clicked from outside is scripted: `dialog.showMessageBox`
// is replaced by a function that records each question and gives the next
// prepared answer, and `Tray` is made to keep its latest menu and tooltip.
// Nothing in src/ knows about this.
//
// What it does, in order, with one run in the middle of a 45 second step:
//
//   1. the tray menu and tooltip say that one agent run is in progress,
//      and the application menu has a Quit of its own (outside macOS, where
//      the system's has it): the tray icon is not always there;
//   2. closing the window asks; "Cancel" leaves everything as it was;
//   3. Quit (the application menu's own, where there is one) asks; "Keep
//      running in the background" hides the windows, and the app and its
//      runtime stay;
//   4. the window, brought back and closed again, asks again, and hides
//      when the answer is the same;
//   5. the run finishes with the window hidden; the tray line goes;
//   6. with another run in a 30 minute step, Quit asks again; "Stop and
//      quit" ends the app by itself, the runtime having been asked to stop
//      the run (its log says so), with nothing left running;
//   7. started again: that run is TERMINATED and stays so, and with nothing
//      running Quit asks nothing and the app is gone in seconds.
//
// Exit code 0 when every check held.

const { spawn } = require("node:child_process");
const fs = require("node:fs");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");

const DESKTOP = path.join(__dirname, "..");
// CountdownTimerBlock (backend/blocks/time_blocks.py), as build/smoke_test.py uses it.
const COUNTDOWN_BLOCK = "d67a9c52-5e4e-11e2-bcfd-0800200c9a71";
const OWNER = { name: "Quit Check", email: "owner@quit.test", password: "password-of-the-owner" };
const READY_MS = 600_000;
const SHORT_STEP_SECONDS = 45;
const LONG_STEP_SECONDS = 1800;
const KEEP_RUNNING = 0;
const STOP_AND_QUIT = 1;
const CANCEL = 2;

const failures = [];
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function expect(what, actual, expected) {
  const held = JSON.stringify(actual) === JSON.stringify(expected);
  console.log(`  ${held ? "ok    " : "FAILED"} ${what} -> ${JSON.stringify(actual)}`);
  if (!held) failures.push(`${what}: got ${JSON.stringify(actual)}, expected ${JSON.stringify(expected)}`);
}

async function until(what, check, timeoutMs = 60_000, everyMs = 500) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    try {
      const found = await check();
      if (found) return found;
    } catch {
      // Not there yet.
    }
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await sleep(everyMs);
  }
}

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

// One request over a debugging WebSocket (Chromium's or Node's: the same
// protocol): evaluate an expression, wait for it if it is a promise, and
// return its value.
async function evaluate(socketUrl, expression) {
  const socket = new WebSocket(socketUrl);
  await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onerror = () => reject(new Error(`cannot connect to ${socketUrl}`));
  });
  try {
    const reply = new Promise((resolve) => {
      socket.onmessage = (message) => {
        const data = JSON.parse(message.data);
        if (data.id === 1) resolve(data);
      };
    });
    socket.send(
      JSON.stringify({
        id: 1,
        method: "Runtime.evaluate",
        params: { expression, returnByValue: true, awaitPromise: true },
      }),
    );
    const { result, error } = await reply;
    if (error) throw new Error(error.message);
    if (result.exceptionDetails) {
      throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    }
    return result.result.value;
  } finally {
    socket.close();
  }
}

class Shell {
  constructor(dataDir) {
    this.dataDir = dataDir;
    this.exit = null;
  }

  async start() {
    this.inspectPort = await freePort();
    this.pagePort = await freePort();
    const log = fs.openSync(path.join(this.dataDir, "shell.log"), "a");
    this.child = spawn(
      require("electron"),
      [
        `--inspect=${this.inspectPort}`,
        DESKTOP,
        `--remote-debugging-port=${this.pagePort}`,
        `--user-data-dir=${path.join(this.dataDir, "electron-profile")}`,
      ],
      {
        // The same start on every machine: no Claude Code sign-in (as the
        // smoke test), and no variant.
        env: {
          ...process.env,
          AUTOGPT_DESKTOP_DATA_DIR: this.dataDir,
          AUTOGPT_DESKTOP_VARIANT: "",
          AUTOGPT_CLAUDE_CODE: "off",
        },
        stdio: ["ignore", log, log],
      },
    );
    this.exit = new Promise((resolve) => this.child.once("exit", (code, signal) => resolve({ code, signal })));
    this.exited = false;
    this.exit.then(() => (this.exited = true));
    const [target] = await until("the main process's debugging port", async () =>
      (await fetch(`http://127.0.0.1:${this.inspectPort}/json/list`)).json(),
    );
    this.mainSocket = target.webSocketDebuggerUrl;
    await this.script();
    this.page = await until(
      "the app's window",
      async () => {
        if (this.exited) throw new Error("the shell exited while starting; see shell.log");
        const pages = await (await fetch(`http://127.0.0.1:${this.pagePort}/json`)).json();
        return pages.find((page) => page.type === "page" && /^http:\/\/127\.0\.0\.1:\d+\//.test(page.url));
      },
      READY_MS,
      1000,
    );
    this.url = new URL(this.page.url).origin;
    // The page has to have loaded before it can be asked to fetch.
    await until("the page to load", () => this.inPage(`document.readyState === "complete"`));
  }

  // In the main process, with Electron's modules as `e`.
  main(body) {
    return evaluate(
      this.mainSocket,
      `(async () => { const e = process.mainModule.require("electron"); ${body} })()`,
    );
  }

  inPage(expression) {
    return evaluate(this.page.webSocketDebuggerUrl, expression);
  }

  // See the top of this file: what cannot be clicked from outside.
  script() {
    return this.main(`
      globalThis.asked = [];
      globalThis.answers = [];
      globalThis.tray = { menu: [], toolTip: null };
      e.dialog.showMessageBox = async (...given) => {
        const question = given[given.length - 1];
        asked.push({ message: question.message, buttons: question.buttons, onAWindow: given.length === 2 });
        return { response: answers.length ? answers.shift() : question.cancelId };
      };
      const { setContextMenu, setToolTip } = e.Tray.prototype;
      e.Tray.prototype.setContextMenu = function (menu) {
        tray.menu = menu.items.map((item) => item.label);
        return setContextMenu.call(this, menu);
      };
      e.Tray.prototype.setToolTip = function (text) {
        tray.toolTip = text;
        return setToolTip.call(this, text);
      };
      return true;
    `);
  }

  answerWith(...answers) {
    return this.main(`answers.push(...${JSON.stringify(answers)}); return true;`);
  }

  state() {
    return this.main(`
      const windows = e.BrowserWindow.getAllWindows();
      return {
        asked: asked.length,
        last: asked[asked.length - 1] || null,
        visible: windows.filter((window) => window.isVisible()).length,
        windows: windows.length,
        trayLine: tray.menu.find((label) => label.includes("in progress")) || null,
        toolTip: tray.toolTip,
      };
    `);
  }

  // The application menu's items that quit the app, as "menu > item".
  quitItems() {
    return this.main(`
      const menus = e.Menu.getApplicationMenu()?.items || [];
      return menus.flatMap((menu) =>
        (menu.submenu?.items || []).filter((item) => item.role === "quit").map((item) => menu.label + " > " + item.label),
      );
    `);
  }

  // As the X on the window does.
  closeWindow() {
    return this.main(`
      const window = e.BrowserWindow.getAllWindows().find((w) => w.webContents.getURL().startsWith(${JSON.stringify(this.url)}));
      window.show();
      window.close();
      return true;
    `);
  }

  // As "Quit" in the tray menu does, and Cmd+Q.
  quit() {
    return this.main(`setImmediate(() => e.app.quit()); return true;`);
  }

  // As choosing Quit in the application menu does (see `quitItems`). False
  // when the menu has none.
  quitFromTheMenu() {
    return this.main(`
      const menus = e.Menu.getApplicationMenu()?.items || [];
      const item = menus.flatMap((menu) => menu.submenu?.items || []).find((item) => item.role === "quit");
      if (!item) return false;
      setImmediate(() => item.click(undefined, e.BrowserWindow.getFocusedWindow(), undefined));
      return true;
    `);
  }

  // The platform's API, called from the window as the app calls it.
  api(apiPath, body) {
    return this.inPage(`(async () => {
      const minted = await (await fetch("/api/auth/token")).json();
      const response = await fetch("/_agpt/api" + ${JSON.stringify(apiPath)}, {
        method: ${body ? '"POST"' : '"GET"'},
        headers: { "Content-Type": "application/json", Authorization: "Bearer " + minted.token },
        ${body ? `body: ${JSON.stringify(JSON.stringify(body))},` : ""}
      });
      return { status: response.status, body: await response.json().catch(() => null) };
    })()`);
  }

  async signIn() {
    const post = (route, body) =>
      this.inPage(`(async () => (await fetch(${JSON.stringify(route)}, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: ${JSON.stringify(JSON.stringify(body))},
      })).status)()`);
    const { name, email, password } = OWNER;
    if ((await post("/api/auth/sign-in/email", { email, password })) === 200) return "signed in";
    return (await post("/api/auth/sign-up/email", { name, email, password })) === 200 ? "signed up" : "refused";
  }

  // An agent of one timer block, running. Returns the path its run is read from.
  async startRun(seconds) {
    const node = {
      block_id: COUNTDOWN_BLOCK,
      input_default: { seconds, input_message: "the timer ran out" },
      metadata: { position: { x: 0, y: 0 } },
    };
    const graph = { name: `Quit check: ${seconds}s`, description: "One block.", nodes: [node], links: [] };
    const created = await this.api("/graphs", { graph });
    if (created.status !== 200) throw new Error(`the agent was not created: ${JSON.stringify(created)}`);
    const { id, version } = created.body;
    const run = await this.api(`/graphs/${id}/execute/${version}`, { inputs: {}, credentials_inputs: {} });
    if (run.status !== 200) throw new Error(`the run was not accepted: ${JSON.stringify(run)}`);
    const runPath = `/graphs/${id}/executions/${run.body.id}`;
    await until(`the run to be in the middle of its step`, async () => (await this.status(runPath)) === "RUNNING", 180_000);
    return runPath;
  }

  async status(runPath) {
    return (await this.api(runPath)).body?.status;
  }

  // Whether the runtime still has its services up (it removes the list of
  // them when it has stopped them).
  servicesRecorded() {
    return fs.existsSync(path.join(this.dataDir, "run", "children.json"));
  }

  async stopWhateverIsLeft() {
    if (!this.child || this.exited) return;
    this.child.kill("SIGKILL");
    // Its runtime sees stdin close and stops the services itself.
    await until("the runtime to stop", () => !this.servicesRecorded(), 120_000, 1000).catch((error) =>
      console.log(String(error)),
    );
  }
}

async function whileARunIsInProgress(shell) {
  console.log("== with a run in the middle of a step");
  const run = await shell.startRun(SHORT_STEP_SECONDS);
  await until("the tray to say so", async () => (await shell.state()).trayLine, 20_000);
  let state = await shell.state();
  expect("the tray menu's line", state.trayLine, "1 agent run in progress");
  expect("the tray icon's tooltip", state.toolTip, "AutoGPT: 1 agent run in progress");
  if (process.platform !== "darwin") {
    expect("the application menu's own Quit", await shell.quitItems(), ["&File > &Quit AutoGPT"]);
  }

  console.log("== closing the window, and answering Cancel");
  await shell.answerWith(CANCEL);
  await shell.closeWindow();
  await until("the question", async () => (await shell.state()).asked === 1, 10_000);
  await sleep(1500);
  state = await shell.state();
  expect("the question", state.last, {
    message: "1 agent run is in progress.",
    buttons: ["Keep running in the background", "Stop and quit AutoGPT", "Cancel"],
    onAWindow: true,
  });
  expect("windows still showing", state.visible, 1);
  expect("the shell has exited", shell.exited, false);

  console.log("== Quit, and answering Keep running in the background");
  await shell.answerWith(KEEP_RUNNING);
  if (process.platform === "darwin") await shell.quit();
  else expect("Quit is chosen in the application menu", await shell.quitFromTheMenu(), true);
  await until("the question", async () => (await shell.state()).asked === 2, 10_000);
  await until("the windows to hide", async () => (await shell.state()).visible === 0, 10_000);
  await sleep(3000);
  state = await shell.state();
  expect("windows showing", state.visible, 0);
  expect("windows kept, hidden", state.windows >= 1, true);
  expect("the shell has exited", shell.exited, false);
  expect("the runtime still lists its services as running", shell.servicesRecorded(), true);
  expect("the run goes on", await shell.status(run), "RUNNING");

  console.log("== the window brought back and closed again, and answering Keep running again");
  await shell.answerWith(KEEP_RUNNING);
  await shell.closeWindow();
  await until("the question, again", async () => (await shell.state()).asked === 3, 10_000);
  await until("the windows to hide", async () => (await shell.state()).visible === 0, 10_000);
  await sleep(2000);
  state = await shell.state();
  expect("questions asked so far (closing it asked again)", state.asked, 3);
  expect("windows showing", state.visible, 0);
  expect("the shell has exited", shell.exited, false);
  expect("the run goes on", await shell.status(run), "RUNNING");

  console.log("== the run finishes with the window hidden");
  await until("the run to finish", async () => (await shell.status(run)) === "COMPLETED", 120_000, 1000);
  expect("the run", await shell.status(run), "COMPLETED");
  await until("the tray line to go", async () => !(await shell.state()).trayLine, 20_000);
  state = await shell.state();
  expect("the tray menu's line", state.trayLine, null);
  expect("the tray icon's tooltip", state.toolTip, "AutoGPT");
  expect("the shell has exited, now that nothing runs", shell.exited, false);
}

async function stopAndQuit(shell) {
  console.log("== another run, Quit, and answering Stop and quit");
  const run = await shell.startRun(LONG_STEP_SECONDS);
  await until("the tray to say so", async () => (await shell.state()).trayLine, 20_000);
  await shell.answerWith(STOP_AND_QUIT);
  const asked = Date.now();
  await shell.quit();
  const { code } = await Promise.race([shell.exit, sleep(120_000).then(() => ({ code: "still running after 120s" }))]);
  const took = (Date.now() - asked) / 1000;
  expect("the shell's exit code", code, 0);
  console.log(`  it took ${took.toFixed(1)}s`);
  expect("...well inside the time a run is waited for (20s) plus a stop", took < 45, true);
  expect("the runtime still lists its services as running", shell.servicesRecorded(), false);
  const log = fs.readFileSync(path.join(shell.dataDir, "logs", "runtime.log"), "utf8");
  expect("the runtime was asked to stop the run, and said how long that may take", /"step":"stopping","message":"Stopping 1 agent run","grace_seconds":\d+/.test(log), true);
  expect("the runtime heard the run stop", log.includes("1 agent run stopped"), true);
  return run;
}

async function afterwards(shell, stopped) {
  console.log("== started again");
  expect("the owner", await shell.signIn(), "signed in");
  expect("the run that was stopped", await shell.status(stopped), "TERMINATED");
  await sleep(20_000);
  expect("...and 20 seconds on", await shell.status(stopped), "TERMINATED");
  const state = await shell.state();
  expect("the tray menu's line", state.trayLine, null);

  console.log("== Quit with nothing running");
  const asked = Date.now();
  await shell.quit();
  const { code } = await Promise.race([shell.exit, sleep(90_000).then(() => ({ code: "still running after 90s" }))]);
  expect("the shell's exit code", code, 0);
  console.log(`  it took ${((Date.now() - asked) / 1000).toFixed(1)}s`);
  expect("the runtime still lists its services as running", shell.servicesRecorded(), false);
  const log = fs.readFileSync(path.join(shell.dataDir, "logs", "runtime.log"), "utf8");
  expect("the runtime was asked to stop a run only once, the first time", log.split('"step":"stopping"').length - 1, 1);
}

async function main() {
  const args = process.argv.slice(2);
  const given = args.includes("--data-dir") ? args[args.indexOf("--data-dir") + 1] : null;
  const dataDir = path.resolve(given || fs.mkdtempSync(path.join(os.tmpdir(), "autogpt-quit-")));
  fs.mkdirSync(dataDir, { recursive: true });
  console.log(`data directory: ${dataDir}`);
  let shell = new Shell(dataDir);
  try {
    await shell.start();
    console.log(`the app is at ${shell.url}`);
    expect("the owner", await shell.signIn(), "signed up");
    await whileARunIsInProgress(shell);
    const stopped = await stopAndQuit(shell);
    shell = new Shell(dataDir);
    await shell.start();
    await afterwards(shell, stopped);
  } catch (error) {
    failures.push(`the check itself failed: ${error.stack || error}`);
  } finally {
    await shell.stopWhateverIsLeft();
    if (!given && !args.includes("--keep")) {
      await until("the data directory to be removable", () => (fs.rmSync(dataDir, { recursive: true, force: true }), true), 30_000).catch(
        (error) => console.log(String(error)),
      );
    }
  }
  if (failures.length) {
    console.log("\nFAILED");
    for (const failure of failures) console.log(`  - ${failure}`);
    process.exitCode = 1;
  } else {
    console.log("\nOK");
  }
}

if (require.main === module) main();

module.exports = { Shell, until, sleep, LONG_STEP_SECONDS };
