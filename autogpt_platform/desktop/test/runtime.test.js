"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { test } = require("node:test");

const { spawn } = require("node:child_process");
const { Runtime, parseEvent, killRecorded } = require("../src/runtime");
const { defaultDataDir, readRuntimeManifest } = require("../src/paths");

const FAKE = path.join(__dirname, "fake-runtime.js");

function collect(runtime) {
  const events = [];
  const logs = [];
  runtime.on("event", (event) => events.push(event));
  runtime.on("log", (line) => logs.push(line));
  return { events, logs };
}

test("parseEvent accepts only JSON objects with an event name", () => {
  assert.deepEqual(parseEvent('{"event":"ready","url":"x"}'), { event: "ready", url: "x" });
  assert.equal(parseEvent("plain text"), null);
  assert.equal(parseEvent('{"no":"event"}'), null);
  assert.equal(parseEvent("{not json"), null);
});

test("runtime reports progress and ready, and stops when stdin closes", async () => {
  const runtime = new Runtime({ command: process.execPath, args: [FAKE] });
  const { events, logs } = collect(runtime);
  const ready = new Promise((resolve) =>
    runtime.on("event", (event) => event.event === "ready" && resolve()),
  );
  const exit = new Promise((resolve) => runtime.once("exit", resolve));
  runtime.start();
  await ready;
  await runtime.stop();
  const { code, expected } = await exit;

  assert.equal(code, 0);
  assert.equal(expected, true);
  assert.deepEqual(
    events.map((event) => event.event),
    ["progress", "ready", "progress"],
  );
  assert.deepEqual(logs, ["a plain log line"]);
});

test("an unexpected exit is reported as not expected", async () => {
  const runtime = new Runtime({ command: process.execPath, args: [FAKE, "crash"] });
  const { events } = collect(runtime);
  const exit = new Promise((resolve) => runtime.once("exit", resolve));
  runtime.start();
  const { code, expected } = await exit;

  assert.equal(code, 3);
  assert.equal(expected, false);
  assert.ok(events.some((event) => event.event === "error" && event.fatal));
});

test("a runtime that ignores stdin is killed after the grace period", async () => {
  const runtime = new Runtime({
    command: process.execPath,
    args: [FAKE, "ignore-stdin"],
    stopGraceMs: 1_500,
  });
  const ready = new Promise((resolve) =>
    runtime.on("event", (event) => event.event === "ready" && resolve()),
  );
  runtime.start();
  await ready;
  const started = Date.now();
  await runtime.stop();
  assert.ok(Date.now() - started >= 1_400);
});

test("a runtime that asks for more time to stop is given it", async () => {
  const runtime = new Runtime({
    command: process.execPath,
    args: [FAKE, "slow-stop"],
    stopGraceMs: 500,
  });
  const ready = new Promise((resolve) =>
    runtime.on("event", (event) => event.event === "ready" && resolve()),
  );
  const exit = new Promise((resolve) => runtime.once("exit", resolve));
  runtime.start();
  await ready;
  await runtime.stop();

  assert.deepEqual(await exit, { code: 0, signal: null, expected: true });
});

function startRunning(...modes) {
  const runtime = new Runtime({
    command: process.execPath,
    args: [FAKE, "running", ...modes],
    stopGraceMs: 500,
  });
  const collected = collect(runtime);
  const said = (text) =>
    new Promise((resolve) =>
      runtime.on("event", (event) => String(event.message).includes(text) && resolve(event)),
    );
  const active = new Promise((resolve) => runtime.on("event", (event) => event.event === "activity" && resolve(event)));
  const exit = new Promise((resolve) => runtime.once("exit", resolve));
  runtime.start();
  return { runtime, ...collected, said, active, exit };
}

test("the runtime says what is running", async () => {
  const { runtime, active } = startRunning();
  assert.deepEqual(await active, { event: "activity", graph_runs: 1, copilot_turns: 0 });
  await runtime.stop();
});

test("stopping the runs is asked for on stdin, which stays open, and is given the time it asks for", async () => {
  const { runtime, events, active, exit } = startRunning();
  await active;
  await runtime.stop({ stopRuns: true });

  // Longer than the 500 ms a stop is allowed here: `grace_seconds` counted.
  assert.deepEqual(await exit, { code: 0, signal: null, expected: true });
  const messages = events.map((event) => event.message).filter(Boolean);
  assert.deepEqual(messages.slice(-2), ['asked: {"command":"stop","drain":true}', "the run was stopped"]);
});

test("a plain stop closes stdin and asks for nothing", async () => {
  const { runtime, events, active } = startRunning();
  await active;
  await runtime.stop();
  assert.equal(events.at(-1).message, "stdin closed");
  assert.ok(!events.some((event) => String(event.message).startsWith("asked")));
});

test("a second stop while the runs are being stopped closes stdin: wait for nothing more", async () => {
  const { runtime, events, active, said, exit } = startRunning("slow");
  await active;
  const asked = said("asked");
  const first = runtime.stop({ stopRuns: true });
  await asked;
  const started = Date.now();
  await runtime.stop({ stopRuns: true });
  await first;

  assert.deepEqual(await exit, { code: 0, signal: null, expected: true });
  assert.equal(events.at(-1).message, "stdin closed");
  assert.equal(events.filter((event) => String(event.message).startsWith("asked")).length, 1);
  assert.ok(Date.now() - started < 5_000);
});

test("stopping the runs of a runtime that has died does not throw", async () => {
  const runtime = new Runtime({ command: process.execPath, args: [FAKE, "crash"] });
  const exit = new Promise((resolve) => runtime.once("exit", resolve));
  runtime.start();
  await exit;
  await runtime.stop({ stopRuns: true });
});

test("a runtime that cannot be started is reported, and stopping it does not hang", async () => {
  const runtime = new Runtime({ command: path.join(__dirname, "no-such-runtime") });
  const { events } = collect(runtime);
  const exit = new Promise((resolve) => runtime.once("exit", resolve));
  runtime.start();

  assert.equal((await exit).expected, false);
  assert.ok(events.some((event) => event.event === "error" && event.fatal));
  await runtime.stop();
});

test("runtime output is appended to the log file", async () => {
  const logFile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "agpt-desktop-")), "runtime.log");
  const runtime = new Runtime({ command: process.execPath, args: [FAKE, "crash"], logFile });
  const exit = new Promise((resolve) => runtime.once("exit", resolve));
  runtime.start();
  await exit;
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.match(fs.readFileSync(logFile, "utf8"), /a plain log line/);
});

test("data directory follows each platform's per-machine location", () => {
  assert.equal(
    defaultDataDir("win32", { LOCALAPPDATA: "C:\\Users\\u\\AppData\\Local" }),
    path.join("C:\\Users\\u\\AppData\\Local", "AutoGPT"),
  );
  assert.equal(
    defaultDataDir("linux", { XDG_DATA_HOME: "/x/share" }),
    path.join("/x/share", "AutoGPT"),
  );
  assert.match(defaultDataDir("darwin", {}), /Library[\\/]Application Support[\\/]AutoGPT$/);
  assert.equal(defaultDataDir("linux", { AUTOGPT_DESKTOP_DATA_DIR: "/custom" }), "/custom");
});

test("manifest commands resolve relative to the runtime directory", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agpt-runtime-"));
  fs.writeFileSync(
    path.join(dir, "manifest.json"),
    JSON.stringify({
      win32: { command: "python/python.exe", args: ["-m", "autogpt_desktop", "{runtime}/x"] },
      default: { command: "python/bin/python3", env: { A: "{runtime}/a" } },
    }),
  );
  const win = readRuntimeManifest(dir, "win32");
  assert.equal(win.command, path.join(dir, "python/python.exe"));
  assert.deepEqual(win.args, ["-m", "autogpt_desktop", `${dir}/x`]);
  const mac = readRuntimeManifest(dir, "darwin");
  assert.equal(mac.command, path.join(dir, "python/bin/python3"));
  assert.deepEqual(mac.env, { A: `${dir}/a` });
});

// The runtime starts each service as the leader of its own process group.
function spawnService(script) {
  return spawn(process.execPath, ["-e", script], {
    stdio: ["ignore", "pipe", "ignore"],
    detached: true,
    windowsHide: true,
  });
}

function writeRegistry(entries) {
  const registry = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "agpt-registry-")), "children.json");
  fs.writeFileSync(registry, JSON.stringify(entries));
  return registry;
}

function isRunning(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function waitUntil(predicate, timeoutMs = 5_000) {
  const deadline = Date.now() + timeoutMs;
  while (!predicate() && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  return predicate();
}

test("services recorded by a runtime that had to be killed are killed too", async () => {
  const service = spawnService("setInterval(() => {}, 1000)");
  const exited = new Promise((resolve) => service.once("exit", (code, signal) => resolve(signal || code)));
  const registry = writeRegistry([{ name: "service", pid: service.pid }, { name: "gone", pid: 999999 }]);

  killRecorded(registry);

  assert.ok(await exited);
  assert.doesNotThrow(() => killRecorded(path.join(os.tmpdir(), "no-such-registry.json")));
});

test(
  "what a recorded service started is killed with it",
  { skip: process.platform === "win32" && "Windows relies on the runtime's Job Object" },
  async () => {
    // Like rabbitmq-server: a launcher that starts the real process and waits.
    const launcher = spawnService(`
      const child = require("node:child_process").spawn(
        process.execPath, ["-e", "setInterval(() => {}, 1000)"], { stdio: "ignore" });
      console.log(child.pid);
      setInterval(() => {}, 1000);
    `);
    const [line] = await require("node:events").once(
      require("node:readline").createInterface({ input: launcher.stdout }),
      "line",
    );
    const started = Number(line);
    assert.ok(isRunning(started));

    killRecorded(writeRegistry([{ name: "launcher", pid: launcher.pid }]));

    assert.ok(await waitUntil(() => !isRunning(started)));
  },
);

test("a runtime that dies takes its recorded services with it", async () => {
  const service = spawnService("setInterval(() => {}, 1000)");
  const exited = new Promise((resolve) => service.once("exit", (code, signal) => resolve(signal || code)));
  const runtime = new Runtime({
    command: process.execPath,
    args: [FAKE, "crash"],
    registryFile: writeRegistry([{ name: "service", pid: service.pid }]),
  });
  runtime.start();

  assert.ok(await exited);
});
