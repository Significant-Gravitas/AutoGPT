"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const { NORMAL, identity } = require("../src/identity");
const { claimDataDir, INSTALL_STAMP } = require("../src/paths");

function fixture(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "autogpt-private-root-"));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const runtime = path.join(directory, "runtime");
  fs.mkdirSync(runtime);
  fs.writeFileSync(path.join(runtime, "manifest.json"), JSON.stringify({
    win32: { command: "python/python.exe", args: ["serve"], env: { UNTRUSTED: "ignored" } },
  }));
  return { directory, runtime, data: path.join(directory, "data") };
}

test("fresh Windows data uses private creation before the ownership stamp", (t) => {
  const f = fixture(t);
  const calls = [];
  const run = (command, args, options) => {
    calls.push({ command, args, options });
    assert.equal(fs.existsSync(f.data), false);
    assert.equal(fs.existsSync(path.join(f.data, INSTALL_STAMP)), false);
    fs.mkdirSync(f.data);
    return { status: 0, signal: null };
  };
  assert.equal(claimDataDir(f.data, NORMAL, { platform: "win32", runtime: f.runtime, run }), NORMAL.dirName);
  assert.equal(calls.length, 1);
  const { command, args, options } = calls[0];
  assert.equal(command, path.join(f.runtime, "python/python.exe"));
  assert.deepEqual(args.slice(0, 4), ["-I", "-S", "-B", "-c"]);
  assert.match(args[4], /autogpt_desktop.shell_data/);
  assert.deepEqual(args.slice(5), [f.runtime, f.data]);
  assert.equal(options.windowsHide, true);
  assert.equal(options.timeout, 15_000);
  assert.equal(options.maxBuffer, 64 * 1024);
  assert.equal(options.cwd, f.runtime);
  assert.equal(options.shell, false);
  assert.equal(options.env.UNTRUSTED, undefined);
  assert.equal(options.env.PYTHONPATH, undefined);
  assert.equal(options.env.ANTHROPIC_API_KEY, undefined);
});

for (const result of [
  { status: 1 }, { status: null, signal: "SIGTERM" },
  { status: 0, error: new Error("injected spawn failure") },
]) {
  test(`private creation failure (${String(result.status)}/${String(result.signal)}) writes no stamp`, (t) => {
    const f = fixture(t);
    assert.throws(() => claimDataDir(f.data, NORMAL, {
      platform: "win32", runtime: f.runtime, run: () => result,
    }), /private data folder/);
    assert.equal(fs.existsSync(f.data), false);
  });
}

test("a concurrent directory creator is not claimed or removed after helper refusal", (t) => {
  const f = fixture(t);
  assert.throws(() => claimDataDir(f.data, NORMAL, {
    platform: "win32", runtime: f.runtime,
    run: () => { fs.mkdirSync(f.data); fs.writeFileSync(path.join(f.data, "marker"), "other"); return { status: 1 }; },
  }), /private data folder/);
  assert.deepEqual(fs.readdirSync(f.data), ["marker"]);
});

test("existing profiles retain their contents and do not run private creation", (t) => {
  const f = fixture(t);
  fs.mkdirSync(f.data);
  fs.writeFileSync(path.join(f.data, "marker"), "kept");
  const options = { platform: "win32", runtime: f.runtime, run: () => assert.fail("existing profile helper") };
  assert.equal(claimDataDir(f.data, NORMAL, options), NORMAL.dirName);
  assert.equal(fs.readFileSync(path.join(f.data, "marker"), "utf8"), "kept");
  const before = fs.readFileSync(path.join(f.data, INSTALL_STAMP));
  assert.equal(claimDataDir(f.data, identity("voice"), options), NORMAL.dirName);
  assert.deepEqual(fs.readFileSync(path.join(f.data, INSTALL_STAMP)), before);
});

test("main turns ownership/preparation errors into a startup refusal", () => {
  const source = fs.readFileSync(path.join(__dirname, "../src/main.js"), "utf8");
  const method = source.slice(source.indexOf("function dataOfAnotherInstall()"), source.indexOf("function startRuntime()"));
  const message = vm.runInNewContext(`${method}\ndataOfAnotherInstall()`, {
    claimDataDir: () => { throw new Error("injected private creation refusal"); },
    dataDir: "/private-test", install: NORMAL, process: { resourcesPath: "/app" },
    app: { isPackaged: true }, runtimeDir: () => "/app/runtime",
  });
  assert.equal(typeof message, "string");
  assert.match(message, /private creation refusal/);
});

function loadMain({ failure = null, owner = NORMAL.dirName, maintenanceFailure = false, isPackaged = false, runtimeOverride } = {}) {
  const source = fs.readFileSync(path.join(__dirname, "../src/main.js"), "utf8");
  const calls = [];
  const maintenanceRuntimes = [];
  let boot;
  const app = {
    isPackaged,
    commandLine: { hasSwitch: () => false },
    setName: () => calls.push("claim-profile"),
    getPath: () => "/app-data",
    setPath: () => {},
    requestSingleInstanceLock: () => { calls.push("instance-lock"); return true; },
    whenReady: () => ({ then: (callback) => { boot = callback; calls.push("when-ready"); } }),
    on: () => {},
    exit: (status) => calls.push(`exit:${status}`),
  };
  const electron = {
    app, ipcMain: { on: () => {}, handle: () => {} },
    dialog: { showErrorBox: (title, message) => calls.push([title, message]) },
  };
  const context = {
    __dirname: path.join(__dirname, "../src"),
    process: { platform: "win32", env: { AUTOGPT_DESKTOP_RUNTIME: runtimeOverride }, resourcesPath: "/bundle" },
    require: (name) => {
      if (name === "electron") return electron;
      if (["./reset-password-window", "./doctor-window"].includes(name)) return {};
      if (name === "./maintenance") return { claimLaunch: (directory, options) => {
        calls.push("lease");
        maintenanceRuntimes.push(options?.runtime);
        if (maintenanceFailure) throw new Error("injected maintenance refusal");
      } };
      if (name === "./paths") return {
        defaultDataDir: () => "/app-data/AutoGPT",
        runtimeDir: (options) => require("../src/paths").runtimeDir({
          ...options, env: { AUTOGPT_DESKTOP_RUNTIME: runtimeOverride },
        }),
        claimDataDir: () => {
          calls.push("private-root-and-stamp");
          if (failure) throw failure;
          return owner;
        },
      };
      return require(name.startsWith(".") ? path.join(__dirname, "../src", name) : name);
    },
  };
  vm.runInNewContext(source, context);
  return { calls, boot, maintenanceRuntimes };
}

for (const [isPackaged, runtimeOverride, expected] of [
  [false, undefined, path.join(__dirname, "../build/runtime")],
  [true, undefined, path.join("/bundle", "runtime")],
  [false, "/custom-runtime", "/custom-runtime"],
  [true, "/custom-runtime", "/custom-runtime"],
]) {
  test(`maintenance uses the shell runtime (packaged: ${isPackaged}, override: ${runtimeOverride})`, () => {
    const { maintenanceRuntimes, boot } = loadMain({ isPackaged, runtimeOverride });
    assert.deepEqual(maintenanceRuntimes, [expected]);
    assert.equal(typeof boot, "function");
  });
}

test("Windows claims a private root before Electron may create a coincident profile", () => {
  const { calls, boot } = loadMain();
  assert.deepEqual(calls, ["lease", "private-root-and-stamp", "claim-profile", "instance-lock", "when-ready"]);
  assert.equal(typeof boot, "function");
});

for (const [name, options] of [
  ["private creation failure", { failure: new Error("injected private root failure") }],
  ["foreign ownership", { owner: "AutoGPT-voice" }],
  ["maintenance refusal", { maintenanceFailure: true }],
]) {
  test(`Windows ${name} never schedules boot, profile writes or runtime startup`, () => {
    const { calls, boot } = loadMain(options);
    assert.equal(boot, undefined);
    assert.equal(calls.includes("claim-profile"), false);
    assert.equal(calls.includes("instance-lock"), false);
    assert.equal(calls.at(-1), "exit:1");
    assert.equal(calls.filter(Array.isArray).length, 1);
  });
}

test("missing Windows runtime fails before creating or claiming a root", (t) => {
  const f = fixture(t);
  fs.unlinkSync(path.join(f.runtime, "manifest.json"));
  assert.throws(() => claimDataDir(f.data, NORMAL, { platform: "win32", runtime: f.runtime }));
  assert.equal(fs.existsSync(f.data), false);
});

test("fresh POSIX roots need no runtime and are created with private mode", (t) => {
  const f = fixture(t);
  assert.equal(claimDataDir(f.data, NORMAL, { platform: "linux", run: () => assert.fail("POSIX helper") }), NORMAL.dirName);
  if (process.platform !== "win32") assert.equal(fs.statSync(f.data).mode & 0o777, 0o700);
});
