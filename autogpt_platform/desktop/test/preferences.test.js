"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { DEFAULT_PREFERENCES, loadPreferences, savePreferences } = require("../src/preferences");

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "autogpt-preferences-test-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return { root, directory: path.join(root, "config"), file: path.join(root, "config", "desktop-preferences.json") };
}

test("missing preferences deny everything without writing a file", (t) => {
  const f = fixture(t);
  assert.deepEqual(loadPreferences(f.root), { version: 1, usageStatistics: false, crashReports: false, launchAtLogin: false, keepRunningInBackground: false });
  assert.deepEqual(fs.readdirSync(f.root), []);
  assert.ok(Object.isFrozen(loadPreferences(f.root)));
});

test("explicit preferences persist atomically and independent changes preserve consent", (t) => {
  const f = fixture(t);
  const enabled = savePreferences(f.root, { usageStatistics: true });
  assert.equal(enabled.usageStatistics, true);
  assert.equal(enabled.crashReports, false);
  savePreferences(f.root, { launchAtLogin: true, keepRunningInBackground: true });
  assert.deepEqual(loadPreferences(f.root), { ...DEFAULT_PREFERENCES, usageStatistics: true, launchAtLogin: true, keepRunningInBackground: true });
  assert.deepEqual(fs.readdirSync(f.directory), ["desktop-preferences.json"]);
  if (process.platform !== "win32") assert.equal(fs.statSync(f.file).mode & 0o777, 0o600);
  savePreferences(f.root, { usageStatistics: false });
  assert.equal(loadPreferences(f.root).usageStatistics, false);
});

test("malformed, partial, future and oversized records fail closed", (t) => {
  const f = fixture(t);
  fs.mkdirSync(f.directory);
  for (const content of ["{", "null", "[]", "true", JSON.stringify({ version: 1, usageStatistics: true }), JSON.stringify({ ...DEFAULT_PREFERENCES, version: 2, usageStatistics: true }), JSON.stringify({ ...DEFAULT_PREFERENCES, crashReports: "true" }), JSON.stringify({ ...DEFAULT_PREFERENCES, unknown: true }), " ".repeat(4097)]) {
    fs.writeFileSync(f.file, content);
    assert.deepEqual(loadPreferences(f.root), DEFAULT_PREFERENCES);
  }
  savePreferences(f.root, { keepRunningInBackground: true });
  assert.deepEqual(loadPreferences(f.root), { ...DEFAULT_PREFERENCES, keepRunningInBackground: true });
});

test("unknown keys and non-booleans never mutate preferences", (t) => {
  const f = fixture(t);
  savePreferences(f.root, { crashReports: true });
  const before = fs.readFileSync(f.file, "utf8");
  for (const patch of [null, [], new Date(), { telemetry: true }, { version: 2 }, { usageStatistics: 1 }, { usageStatistics: "false" }, { crashReports: undefined }, JSON.parse('{"__proto__":true}'), { [Symbol("hidden")]: true }]) {
    assert.throws(() => savePreferences(f.root, patch), TypeError);
    assert.equal(fs.readFileSync(f.file, "utf8"), before);
  }
});

test("failed publication retains the old consent and removes only the temporary file", (t) => {
  const f = fixture(t);
  savePreferences(f.root, { usageStatistics: true });
  const before = fs.readFileSync(f.file, "utf8");
  t.mock.method(fs, "renameSync", () => { throw new Error("simulated disk failure"); });
  assert.throws(() => savePreferences(f.root, { usageStatistics: false }), /simulated disk failure/);
  assert.equal(fs.readFileSync(f.file, "utf8"), before);
  assert.deepEqual(fs.readdirSync(f.directory), ["desktop-preferences.json"]);
});

test("configuration-directory links cannot import or overwrite another profile's consent", (t) => {
  const f = fixture(t);
  const other = path.join(f.root, "other");
  fs.mkdirSync(other);
  const outside = path.join(other, "desktop-preferences.json");
  const contents = JSON.stringify({ ...DEFAULT_PREFERENCES, usageStatistics: true });
  fs.writeFileSync(outside, contents);
  try {
    fs.symlinkSync(other, f.directory, process.platform === "win32" ? "junction" : "dir");
  } catch (error) {
    if (error.code === "EPERM") return t.skip("symlinks unavailable");
    throw error;
  }
  assert.deepEqual(loadPreferences(f.root), DEFAULT_PREFERENCES);
  assert.throws(() => savePreferences(f.root, { crashReports: true }), /regular configuration directory/);
  assert.equal(fs.readFileSync(outside, "utf8"), contents);
});
