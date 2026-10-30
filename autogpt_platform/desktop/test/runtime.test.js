"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { test } = require("node:test");
const { EventEmitter } = require("node:events");
const { PassThrough, Writable } = require("node:stream");
const vm = require("node:vm");

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

test("spawn failure leaves an old registry untouched and never signals its PIDs", async (t) => {
  const signals = [];
  t.mock.method(process, "kill", (...args) => signals.push(args));
  const registry = writeRegistry([{ name: "old service", pid: process.pid, started: 1 }]);
  const original = fs.readFileSync(registry);
  const runtime = new Runtime({
    command: path.join(__dirname, "no-such-runtime"),
    registryFile: registry,
  });
  runtime.start();
  await runtime.exited;
  assert.deepEqual(signals, []);
  assert.deepEqual(fs.readFileSync(registry), original);
});

test("the shell cannot reap a registry without an identity-checking runtime helper", async (t) => {
  const signals = [];
  t.mock.method(process, "kill", (...args) => signals.push(args));
  const registry = writeRegistry([{ name: "reused PID", pid: process.pid, started: 1 }]);
  const original = fs.readFileSync(registry);
  const cleanup = killRecorded(registry);
  assert.deepEqual(signals, []);
  await assert.rejects(cleanup, /runtime|helper/i);
  assert.deepEqual(fs.readFileSync(registry), original);
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

function writeRegistry(entries) {
  const registry = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "agpt-registry-")), "children.json");
  fs.writeFileSync(registry, JSON.stringify(entries));
  return registry;
}

function controlledRuntime(spawn, log) {
  const module = { exports: {} };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../src/runtime.js"), "utf8"), {
    require(name) {
      if (name === "node:child_process") return { spawn };
      if (name === "node:fs" && log) return { ...fs, createWriteStream: () => log };
      return require(name);
    },
    module, process, setTimeout, clearTimeout,
  });
  return module.exports;
}

test("recorded services are delegated to the bounded Python identity-checking reaper", async () => {
  const registry = writeRegistry([{ name: "service", pid: 123, started: 456 }]);
  const env = { SAFE: "value" };
  const { killRecorded: reap } = controlledRuntime((command, args, options) => {
    assert.equal(command, "bundled-python");
    assert.deepEqual(Array.from(args), ["-B", "-m", "autogpt_desktop.reap", "--data-dir", path.dirname(path.dirname(registry))]);
    assert.equal(options.cwd, "runtime-root");
    assert.equal(options.env, env);
    assert.equal(options.timeout, 15_000);
    assert.equal(options.killSignal, "SIGKILL");
    assert.equal(options.windowsHide, true);
    assert.equal(options.stdio, "ignore");
    const helper = new EventEmitter();
    queueMicrotask(() => helper.emit("close", 0, null));
    return helper;
  });
  await reap(registry, { command: "bundled-python", cwd: "runtime-root", env });
  assert.ok(fs.existsSync(registry), "only the verified helper may remove the registry");
  await reap(path.join(os.tmpdir(), "no-such-registry.json"));
});

for (const outcome of ["spawn-error", "nonzero", "timeout"]) {
  test(`cleanup ${outcome} retains the registry without directly killing processes`, async (t) => {
    t.mock.method(process, "kill", () => assert.fail("unverified PID kill"));
    const registry = writeRegistry([{ name: "service", pid: 123, started: 456 }]);
    const original = fs.readFileSync(registry);
    const { killRecorded: reap } = controlledRuntime(() => {
      const helper = new EventEmitter();
      queueMicrotask(() => {
        if (outcome === "spawn-error") helper.emit("error", new Error("missing helper"));
        else helper.emit("close", outcome === "nonzero" ? 1 : null, outcome === "timeout" ? "SIGKILL" : null);
      });
      return helper;
    });
    await assert.rejects(reap(registry, { command: "bundled-python" }));
    assert.deepEqual(fs.readFileSync(registry), original);
  });
}

test("runtime exit and stop wait for verified cleanup, once only", async () => {
  const registry = writeRegistry([{ name: "service", pid: 123, started: 456 }]);
  const child = Object.assign(new EventEmitter(), {
    pid: 1234, stdin: new PassThrough(), stdout: new PassThrough(), stderr: new PassThrough(),
  });
  const helper = new EventEmitter();
  let spawns = 0;
  const { Runtime: Controlled } = controlledRuntime(() => ++spawns === 1 ? child : helper);
  const runtime = new Controlled({ command: "bundled-python", registryFile: registry });
  let exited = false;
  runtime.on("exit", () => { exited = true; });
  runtime.start();
  child.emit("exit", 3, null);
  child.emit("exit", 3, null);
  let stopped = false;
  const stop = runtime.stop().then(() => { stopped = true; });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(exited, false);
  assert.equal(stopped, false);
  assert.equal(spawns, 2);
  helper.emit("close", 0, null);
  await stop;
  assert.equal(exited, true);
  assert.equal(stopped, true);
  for (const stream of [child.stdin, child.stdout, child.stderr]) stream.end();
});


function logFixture(t, { registry, failWrite = false } = {}) {
  const child = Object.assign(new EventEmitter(), {
    pid: 1234, stdin: new PassThrough(), stdout: new PassThrough(), stderr: new PassThrough(),
  });
  const chunks = [];
  const sink = new Writable({ write(chunk, encoding, callback) {
    chunks.push(chunk.toString());
    callback(failWrite ? Object.assign(new Error("disk unavailable"), { code: "EIO" }) : null);
  } });
  const helper = new EventEmitter();
  let spawned = 0;
  const { Runtime: Controlled } = controlledRuntime(() => ++spawned === 1 ? child : helper, sink);
  const runtime = new Controlled({ command: "fixture", logFile: "fixture.log", registryFile: registry });
  const collected = collect(runtime);
  t.after(() => {
    for (const stream of [child.stdin, child.stdout, child.stderr, sink]) stream.destroy();
  });
  runtime.start();
  return { child, chunks, sink, helper, runtime, ...collected };
}

test("buffered child output after exit is logged before stdio closes", async (t) => {
  const f = logFixture(t);
  f.child.stdout.write("before exit\n");
  f.child.emit("exit", 3, null);
  await f.runtime.exited;
  assert.equal(f.sink.writableEnded, false);
  f.child.stdout.write("buffered stdout\n");
  f.child.stderr.write("buffered stderr\n");
  f.child.emit("close", 3, null);
  assert.equal(f.sink.writableEnded, true);
  assert.deepEqual(f.chunks, ["before exit\n", "buffered stdout\n", "buffered stderr\n"]);
  f.child.stdout.write("late fixture line\n");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(f.chunks.length, 3);
  assert.ok(f.logs.includes("late fixture line"));
});

test("stdio close waits for reaper diagnostics before ending the log", async (t) => {
  const f = logFixture(t, { registry: writeRegistry([]) });
  f.child.emit("exit", 3, null);
  f.child.emit("close", 3, null);
  assert.equal(f.sink.writableEnded, false);
  f.helper.emit("close", 1, null);
  await f.runtime.exited;
  assert.equal(f.sink.writableEnded, true);
  assert.match(f.chunks.join(""), /Runtime cleanup could not complete/);
});

test("file write errors cannot interrupt child output or exit delivery", async (t) => {
  const f = logFixture(t, { failWrite: true });
  f.child.stdout.write("first line\n");
  await new Promise((resolve) => setImmediate(resolve));
  f.child.stderr.write("after failure\n");
  f.child.emit("exit", 3, null);
  f.child.emit("close", 3, null);
  const result = await f.runtime.exited;
  assert.equal(result.code, 3);
  assert.ok(f.logs.includes("after failure"));
  assert.equal(f.logs.filter((line) => line === "Runtime file logging is unavailable.").length, 1);
});
