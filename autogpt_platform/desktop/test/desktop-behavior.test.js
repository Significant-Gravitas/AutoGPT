"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const { backgroundOnClose, loginAvailability, loginOptions, applyPreferences } = require("../src/desktop-behavior");

test("background preference never intercepts Quit, shutdown, or failed runtime", () => {
  const base = { keepRunningInBackground: true, quitting: false, sessionEnding: false, runtimeExited: false };
  assert.equal(backgroundOnClose(base), true);
  for (const key of ["quitting", "sessionEnding", "runtimeExited"]) {
    assert.equal(backgroundOnClose({ ...base, [key]: true }), false);
  }
  assert.equal(backgroundOnClose({ ...base, keepRunningInBackground: false }), false);
});

test("login launch refuses development and custom profiles that cannot be reopened by the OS", () => {
  const base = { isPackaged: true, platform: "win32", customProfile: false };
  assert.equal(loginAvailability(base).supported, true);
  assert.equal(loginAvailability({ ...base, platform: "darwin" }).supported, true);
  for (const patch of [{ isPackaged: false }, { customProfile: true }, { platform: "linux" }]) {
    assert.equal(loginAvailability({ ...base, ...patch }).supported, false);
  }
  const options = loginOptions(true, "win32", "C:\\Apps\\O'Brien\\AutoGPT.exe", "AutoGPT-test");
  assert.equal(options.path, "C:\\Apps\\O'Brien\\AutoGPT.exe");
  assert.equal(options.name, "AutoGPT-test");
  assert.deepEqual(options.args, []);
  assert.deepEqual(loginOptions(false, "darwin", "/Applications/AutoGPT.app", "AutoGPT"), { openAtLogin: false });
});

test("a failed preferences save restores the prior login registration", async () => {
  const calls = [];
  await assert.rejects(applyPreferences({
    current: { launchAtLogin: false }, patch: { launchAtLogin: true },
    supported: true, save: () => { throw new Error("disk full"); },
    setLogin: enabled => calls.push(enabled),
  }), /disk full/);
  assert.deepEqual(calls, [true, false]);
});

test("unavailable login launch and invalid preference types cannot reach native APIs", async () => {
  for (const patch of [{ launchAtLogin: true }, { launchAtLogin: "true" }, { arbitrary: true }]) {
    let touched = false;
    await assert.rejects(applyPreferences({ current: { launchAtLogin: false }, patch, supported: false,
      save: () => { touched = true; }, setLogin: () => { touched = true; } }));
    assert.equal(touched, false);
  }
});

test("privacy and background choices do not modify OS startup registration", async () => {
  const patch = { usageStatistics: false, crashReports: true, keepRunningInBackground: true };
  const next = await applyPreferences({ current: { launchAtLogin: false }, patch, supported: false,
    save: input => ({ launchAtLogin: false, ...input }), setLogin: () => assert.fail("unexpected login change") });
  assert.deepEqual(next, { launchAtLogin: false, ...patch });
});

test("saving a login choice reconciles OS changes even when the saved preference matches", async () => {
  for (const enabled of [true, false]) {
    let actual = !enabled;
    await applyPreferences({ current: { launchAtLogin: enabled }, patch: { launchAtLogin: enabled }, supported: true,
      getLogin: () => actual, setLogin: value => { actual = value; }, save: patch => patch });
    assert.equal(actual, enabled);
  }
});

test("failed save restores observed OS registration rather than a stale preference", async () => {
  const calls = [];
  await assert.rejects(applyPreferences({ current: { launchAtLogin: false }, patch: { launchAtLogin: false }, supported: true,
    getLogin: () => true, setLogin: value => calls.push(value), save: () => { throw new Error("disk full"); } }), /disk full/);
  assert.deepEqual(calls, [false, true]);
});
