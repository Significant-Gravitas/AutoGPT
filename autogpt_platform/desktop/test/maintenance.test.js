"use strict";

const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { canonicalDirectory, claimLaunch: claimLaunchWithProbe, maintenancePaths, operationActive, processStartProbe } = require("../src/maintenance");
const { claimDataDir } = require("../src/paths");
const { NORMAL } = require("../src/identity");

function claimLaunch(directory, options = {}) {
  return claimLaunchWithProbe(directory, { processStarted: () => 12345, ...options });
}

function fixture(t, alias = false) {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "autogpt-launch-test-"));
  t.after(() => fs.rmSync(temporary, { recursive: true, force: true }));
  let parent = temporary;
  if (alias) {
    const target = path.join(temporary, "target");
    parent = path.join(temporary, "alias");
    fs.mkdirSync(target);
    fs.symlinkSync(target, parent, process.platform === "win32" ? "junction" : "dir");
  }
  const data = path.join(parent, "AutoGPT");
  return { temporary: parent, data, ...maintenancePaths(data) };
}

test("a launch publishes its live lease before the data directory exists", (t) => {
  const f = fixture(t);
  const lease = claimLaunch(f.data);
  assert.equal(JSON.parse(fs.readFileSync(lease, "utf8")).pid, process.pid);
  assert.equal(fs.existsSync(f.data), false);
  assert.equal(claimDataDir(f.data, NORMAL, { platform: "linux" }), NORMAL.dirName);
  assert.equal(fs.existsSync(lease), true);
});

test("maintenance blocks launch without recreating the root during a restore swap", (t) => {
  const f = fixture(t);
  fs.writeFileSync(f.operation, JSON.stringify({ pid: process.pid }));
  assert.throws(() => { claimLaunch(f.data); claimDataDir(f.data, NORMAL); }, /backup or restore is in progress/);
  assert.equal(fs.existsSync(f.data), false);
  assert.equal(fs.readdirSync(f.temporary).filter((name) => name.startsWith(".autogpt-shell-")).length, 0);
});

test("interrupted restore blocks startup even after its process exits", (t) => {
  const f = fixture(t);
  fs.writeFileSync(f.journal, "{}");
  assert.throws(() => claimLaunch(f.data), /interrupted restore needs recovery/);
  assert.equal(fs.existsSync(f.data), false);
});

test("invalid operation records fail closed", (t) => {
  const f = fixture(t);
  for (const content of ["", "{", "{}", "null", '{"pid":0}', '{"pid":-1}', '{"pid":"123"}']) {
    fs.writeFileSync(f.operation, content);
    assert.equal(operationActive(f.operation), true);
    assert.throws(() => claimLaunch(f.data), /backup or restore is in progress/);
  }
});

test("a dead maintenance process does not prevent a new launch", (t) => {
  const f = fixture(t);
  const child = spawnSync(process.execPath, ["-e", "process.stdout.write(String(process.pid))"], { encoding: "utf8" });
  assert.equal(child.status, 0);
  fs.writeFileSync(f.operation, JSON.stringify({ pid: Number(child.stdout) }));
  assert.equal(operationActive(f.operation), false);
  assert.ok(fs.existsSync(claimLaunch(f.data)));
});

test("a missing root below an alias uses the same maintenance key", (t) => {
  const f = fixture(t);
  const target = path.join(f.temporary, "target");
  const alias = path.join(f.temporary, "alias");
  fs.mkdirSync(target);
  try {
    fs.symlinkSync(target, alias, process.platform === "win32" ? "junction" : "dir");
  } catch (error) {
    if (error.code === "EPERM") return t.skip("symlinks unavailable");
    throw error;
  }
  assert.deepEqual(maintenancePaths(path.join(alias, "missing")), maintenancePaths(path.join(target, "missing")));
  fs.rmdirSync(target);
  assert.deepEqual(maintenancePaths(alias), maintenancePaths(target));
});

test("Darwin identity agrees across case and Unicode normalization", (t) => {
  const f = fixture(t);
  assert.equal(canonicalDirectory(path.join(f.temporary, "CAFÉ"), "darwin"), canonicalDirectory(path.join(f.temporary, "cafe\u0301"), "darwin"));
});

test("shell claims maintenance exclusion before profile and single-instance writes", () => {
  const source = fs.readFileSync(path.join(__dirname, "../src/main.js"), "utf8");
  const lease = source.indexOf("claimLaunch(dataDir)");
  assert.ok(lease > 0);
  assert.ok(lease < source.indexOf("claim(app, install)"));
  assert.ok(lease < source.indexOf("app.requestSingleInstanceLock()"));
});

for (const alias of [false, true]) test(`new launches remove only confirmed dead leases (aliased root: ${alias})`, (t) => {
  const f = fixture(t, alias);
  const prefix = `.autogpt-shell-${f.key}-`;
  const records = new Map([
    [`${prefix}dead.json`, '{"pid":234}'],
    [`${prefix}live.json`, JSON.stringify({ pid: process.pid })],
    [`${prefix}denied.json`, '{"pid":345}'],
    [`${prefix}invalid.json`, '{"pid":"234"}'],
    [`${prefix}malformed.json`, '{'],
    [`${prefix}unreadable.json`, '{"pid":234}'],
    [`${prefix}pending.json.tmp`, '{"pid":234}'],
    ['.autogpt-shell-other-dead.json', '{"pid":234}'],
  ]);
  for (const [name, value] of records) fs.writeFileSync(path.join(f.temporary, name), value);
  const read = fs.readFileSync;
  t.mock.method(fs, "readFileSync", (file, ...args) => {
    if (typeof file === "string" && path.basename(file) === `${prefix}unreadable.json`) {
      throw Object.assign(new Error("denied"), { code: "EACCES" });
    }
    return read(file, ...args);
  });
  t.mock.method(process, "kill", (pid, signal) => {
    assert.equal(signal, 0);
    if (pid === 234) throw Object.assign(new Error("gone"), { code: "ESRCH" });
    if (pid === 345) throw Object.assign(new Error("denied"), { code: "EPERM" });
    assert.equal(pid, process.pid);
  });
  const lease = claimLaunch(f.data);
  assert.equal(fs.existsSync(path.join(f.temporary, `${prefix}dead.json`)), false);
  for (const name of records.keys()) {
    if (name !== `${prefix}dead.json`) assert.equal(fs.existsSync(path.join(f.temporary, name)), true, name);
  }
  assert.equal(fs.existsSync(lease), true);
});

test("an unreadable lease inventory does not prevent publishing a new lease", (t) => {
  const f = fixture(t);
  t.mock.method(fs, "readdirSync", () => { throw Object.assign(new Error("denied"), { code: "EACCES" }); });
  assert.equal(fs.existsSync(claimLaunch(f.data)), true);
});


test("process birth distinguishes a reused operation PID", (t) => {
  const f = fixture(t);
  fs.writeFileSync(f.operation, JSON.stringify({ pid: process.pid, started: 100 }));
  assert.equal(operationActive(f.operation, () => 100), true);
  assert.equal(operationActive(f.operation, () => 101), false);
  assert.equal(operationActive(f.operation, () => null), false);
  assert.equal(operationActive(f.operation, () => { throw new Error("denied"); }), true);
});

test("launch records native birth and prunes reused shell identities", (t) => {
  const f = fixture(t);
  const stale = path.join(f.temporary, `.autogpt-shell-${f.key}-old.json`);
  fs.writeFileSync(stale, JSON.stringify({ pid: process.pid, started: 99 }));
  const lease = claimLaunch(f.data, { processStarted: () => 100 });
  assert.deepEqual(JSON.parse(fs.readFileSync(lease, "utf8")), { pid: process.pid, started: 100 });
  assert.equal(fs.existsSync(stale), false);
});

test("unknown current identity refuses launch before publishing its lease", (t) => {
  const f = fixture(t);
  for (const value of [null, true, 0, NaN, Infinity, "100"]) {
    assert.throws(() => claimLaunch(f.data, { processStarted: () => value }), /identity/);
  }
  assert.equal(fs.readdirSync(f.temporary).length, 0);
});


test("invalid marker identities never invoke native inspection", (t) => {
  const f = fixture(t);
  for (const pid of [0, -1, true, "123", 2 ** 31]) {
    fs.writeFileSync(f.operation, JSON.stringify({ pid, started: 100 }));
    assert.equal(operationActive(f.operation, () => assert.fail("invalid PID inspected")), true);
  }
  for (const started of [null, true, "100", -1, 0]) {
    fs.writeFileSync(f.operation, JSON.stringify({ pid: process.pid, started }));
    assert.equal(operationActive(f.operation, () => assert.fail("legacy identity inspected")), true);
  }
});

test("native identity probe is isolated, bounded and validates its result", (t) => {
  const f = fixture(t);
  const runtime = path.join(f.temporary, "runtime");
  fs.mkdirSync(runtime);
  fs.writeFileSync(path.join(runtime, "manifest.json"), JSON.stringify({
    default: { command: "python/bin/python3", args: ["UNTRUSTED"], env: { SHOULD_NOT_PASS: "value" } },
  }));
  let response = { status: 0, signal: null, stdout: "100.25\n" };
  let calls = 0;
  const probe = processStartProbe({ runtime, platform: "linux", run(command, args, options) {
    calls += 1;
    assert.equal(command, path.join(runtime, "python/bin/python3"));
    assert.deepEqual(args.slice(0, 3), ["-I", "-B", "-c"]);
    assert.match(args[3], /process_started/);
    assert.equal(args.at(-1), "123");
    assert.equal(args.at(-2), runtime);
    assert.equal(options.cwd, runtime);
    assert.ok(options.timeout > 0 && options.timeout <= 15000);
    assert.equal(options.shell, false);
    assert.equal(options.windowsHide, true);
    assert.equal(options.maxBuffer, 4096);
    assert.ok(Object.keys(options.env).every((name) => ["windir", "systemroot"].includes(name.toLowerCase())));
    return response;
  } });
  assert.equal(probe(123), 100.25);
  response = { status: 0, signal: null, stdout: "null" };
  assert.equal(probe(123), null);
  for (const invalid of ["{}", "[]", "true", "0", "-1", '"100"', "NaN", "private output"]) {
    response = { status: 0, signal: null, stdout: invalid };
    assert.throws(() => probe(123));
  }
  for (const failure of [{ status: 1 }, { status: 0, signal: "SIGTERM" }, { status: 0, error: new Error("timeout") }]) {
    response = { stdout: "100.25", ...failure };
    assert.throws(() => probe(123), /identity/);
  }
  const before = calls;
  t.mock.method(performance, "now", () => Number.MAX_SAFE_INTEGER);
  assert.throws(() => probe(123), /timed out/);
  assert.equal(calls, before);
});

test("an active maintenance marker is checked after lease publication", (t) => {
  const f = fixture(t);
  fs.writeFileSync(f.operation, JSON.stringify({ pid: process.pid, started: 100 }));
  let calls = 0;
  assert.throws(() => claimLaunch(f.data, { processStarted() {
    calls += 1;
    if (calls === 2) {
      assert.equal(fs.readdirSync(f.temporary).filter((name) => name.startsWith(".autogpt-shell-")).length, 1);
    }
    return 100;
  } }), /backup or restore/);
  assert.equal(calls, 2);
  assert.equal(fs.readdirSync(f.temporary).filter((name) => name.startsWith(".autogpt-shell-")).length, 0);
});
