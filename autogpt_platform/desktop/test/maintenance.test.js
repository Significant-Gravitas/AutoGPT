"use strict";

const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { canonicalDirectory, claimLaunch, maintenancePaths, operationActive } = require("../src/maintenance");
const { claimDataDir } = require("../src/paths");
const { NORMAL } = require("../src/identity");

function fixture(t) {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "autogpt-launch-test-"));
  t.after(() => fs.rmSync(temporary, { recursive: true, force: true }));
  const data = path.join(temporary, "AutoGPT");
  return { temporary, data, ...maintenancePaths(data) };
}

test("a launch publishes its live lease before the data directory exists", (t) => {
  const f = fixture(t);
  const lease = claimLaunch(f.data);
  assert.equal(JSON.parse(fs.readFileSync(lease, "utf8")).pid, process.pid);
  assert.equal(fs.existsSync(f.data), false);
  assert.equal(claimDataDir(f.data, NORMAL), NORMAL.dirName);
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
