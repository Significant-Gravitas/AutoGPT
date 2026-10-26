"use strict";

const assert = require("node:assert/strict");
const { EventEmitter } = require("node:events");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");

const {
  CHECK_EVERY_MS,
  FIRST_CHECK_MS,
  RECHECK_MS,
  chooseAutoUpdater,
  createUpdater,
  dropsUpdaterQuit,
  failedStartOffer,
  installsFromDownloadedFile,
  nextState,
  releaseUrl,
  updateMenuItems,
  updateMode,
  withUpdatesMenu,
} = require("../src/updater");

// These tests are of the normal app, whatever variant the environment they
// run in names (variants.test.js has the variants).
delete process.env.AUTOGPT_DESKTOP_VARIANT;

const RELEASED = { version: "1.4.0", isPackaged: true };

// electron-updater's autoUpdater, as far as updater.js uses it.
class FakeAutoUpdater extends EventEmitter {
  constructor() {
    super();
    // What electron-updater's constructor decides for a version like 1.4.0-rc.1.
    this.allowPrerelease = true;
    this.autoDownload = true;
    this.autoInstallOnAppQuit = true;
    this.checks = 0;
    this.downloads = 0;
    this.installs = [];
    this.checkResult = () => Promise.resolve(null);
    this.downloadResult = () => Promise.resolve([]);
  }

  checkForUpdates() {
    this.checks += 1;
    return this.checkResult();
  }

  downloadUpdate() {
    this.downloads += 1;
    return this.downloadResult();
  }

  quitAndInstall(...args) {
    this.installs.push(args);
  }
}

function fakeTimers() {
  const scheduled = [];
  const timer = { unref() {} };
  return {
    scheduled,
    setTimeout: (callback, delay) => scheduled.push({ callback, delay, repeats: false }) && timer,
    setInterval: (callback, delay) => scheduled.push({ callback, delay, repeats: true }) && timer,
  };
}

function started(mode = "install", options = {}) {
  const autoUpdater = new FakeAutoUpdater();
  const timers = fakeTimers();
  const states = [];
  const onChange = (state) => states.push(state);
  const updater = createUpdater({ autoUpdater, mode, timers, onChange, ...options });
  return { autoUpdater, timers, states, updater };
}

// An updater that has downloaded 1.5.0 to `file`.
function ready(options = {}, file = "/cache/pending/AutoGPT-Setup-1.5.0-x64.exe") {
  const all = started("install", options);
  all.autoUpdater.emit("update-available", { version: "1.5.0" });
  all.autoUpdater.emit("update-downloaded", { version: "1.5.0", downloadedFile: file });
  return all;
}

const tick = () => new Promise((resolve) => setImmediate(resolve));

test("a development build never looks for updates", () => {
  for (const platform of ["win32", "darwin", "linux"]) {
    assert.equal(updateMode({ version: "0.0.0-dev.0", isPackaged: true, platform }), "off");
    assert.equal(updateMode({ version: "0.0.0-dev.412", isPackaged: true, platform }), "off");
    // The build the upgrade test installs over the first one (desktop-build.yml).
    assert.equal(updateMode({ version: "0.0.1-dev.412", isPackaged: true, platform }), "off");
    assert.equal(updateMode({ version: "1.4.0", isPackaged: false, platform }), "off");
  }
});

test("the version in package.json is a development version", () => {
  const { version } = require("../package.json");
  assert.equal(updateMode({ version, isPackaged: true, platform: "win32" }), "off");
});

test("a release that is itself a pre-release still follows the stable releases", () => {
  assert.equal(updateMode({ version: "1.4.0-rc.1", isPackaged: true, platform: "win32" }), "install");
});

test("updates can be switched off from the environment", () => {
  const env = { AUTOGPT_DESKTOP_UPDATES: "off" };
  assert.equal(updateMode({ ...RELEASED, platform: "win32", env }), "off");
});

test("Windows and an AppImage update in place", () => {
  assert.equal(updateMode({ ...RELEASED, platform: "win32" }), "install");
  const env = { APPIMAGE: "/home/me/Applications/AutoGPT.AppImage" };
  assert.equal(updateMode({ ...RELEASED, platform: "linux", env }), "install");
});

test("a .deb only says that a version is available", () => {
  assert.equal(updateMode({ ...RELEASED, platform: "linux", env: {} }), "notify");
});

test("macOS updates in place only when signed with a Developer ID", () => {
  const appPath = "/Applications/AutoGPT.app/Contents/Resources/app.asar";
  assert.equal(updateMode({ ...RELEASED, platform: "darwin", appPath }), "notify");
  assert.equal(updateMode({ ...RELEASED, platform: "darwin", appPath, macDeveloperId: true }), "install");
});

test("macOS cannot replace an app that runs from the disk image or was never moved", () => {
  for (const appPath of [
    "/Volumes/AutoGPT 1.4.0-arm64/AutoGPT.app/Contents/Resources/app.asar",
    "/private/var/folders/x1/T/AppTranslocation/6E0A/d/AutoGPT.app/Contents/Resources/app.asar",
  ]) {
    assert.equal(updateMode({ ...RELEASED, platform: "darwin", appPath, macDeveloperId: true }), "notify");
  }
});

test("the release page of a version is its desktop-v tag", () => {
  assert.equal(releaseUrl("1.4.0"), "https://github.com/ntindle/autogpt/releases/tag/desktop-v1.4.0");
});

test("the app looks at the repository the installers say they come from", () => {
  const config = require("../electron-builder.config.js");
  const [{ provider, owner, repo }] = config.publish;
  assert.equal(provider, "github");
  assert.ok(releaseUrl("1.4.0").startsWith(`https://github.com/${owner}/${repo}/releases/`));
});

test("a version that merely has dev in it is a release", () => {
  assert.equal(updateMode({ version: "1.4.0-dev.1", isPackaged: true, platform: "win32" }), "install");
  assert.equal(updateMode({ version: "0.1.0", isPackaged: true, platform: "win32" }), "install");
});

test("an AppImage uses the AppImage updater whatever package-type says", () => {
  class AppImageUpdater {}
  const electronUpdater = {
    AppImageUpdater,
    get autoUpdater() {
      return "the one electron-updater picks";
    },
  };
  const env = { APPIMAGE: "/home/me/Applications/AutoGPT.AppImage" };
  assert.ok(chooseAutoUpdater(electronUpdater, { platform: "linux", env }) instanceof AppImageUpdater);
  assert.equal(chooseAutoUpdater(electronUpdater, { platform: "linux", env: {} }), "the one electron-updater picks");
  assert.equal(chooseAutoUpdater(electronUpdater, { platform: "win32", env }), "the one electron-updater picks");
  assert.equal(chooseAutoUpdater(electronUpdater, { platform: "darwin", env: {} }), "the one electron-updater picks");
});

test("only what electron-updater verified is ever installed, and only when asked", () => {
  const { autoUpdater } = started("install");
  // Downloads are started by updater.js, through electron-updater's own
  // downloadUpdate, which verifies the file.
  assert.equal(autoUpdater.autoDownload, false);
  assert.equal(autoUpdater.autoInstallOnAppQuit, false);
  assert.equal(autoUpdater.allowDowngrade, false);
  assert.equal(autoUpdater.disableWebInstaller, true);
  // Never set: assigning a channel switches allowDowngrade on.
  assert.equal(autoUpdater.channel, undefined);
});

test("a pre-release version does not send electron-updater looking for pre-release tags", () => {
  const { autoUpdater } = started("install");
  assert.equal(autoUpdater.allowPrerelease, false);
});

test("where the app cannot install, nothing is downloaded", () => {
  const { autoUpdater, updater } = started("notify");
  assert.equal(autoUpdater.autoDownload, false);
  autoUpdater.emit("update-available", { version: "1.5.0" });
  assert.equal(updater.state().phase, "available");
  assert.equal(autoUpdater.downloads, 0);
});

test("nothing is checked until the runtime is ready, then after a minute and every six hours", () => {
  const { autoUpdater, timers, updater } = started();
  assert.equal(timers.scheduled.length, 0);
  updater.arm();
  updater.arm();
  assert.deepEqual(
    timers.scheduled.map(({ delay, repeats }) => ({ delay, repeats })),
    [
      { delay: FIRST_CHECK_MS, repeats: false },
      { delay: CHECK_EVERY_MS, repeats: true },
    ],
  );
  assert.equal(autoUpdater.checks, 0);
  timers.scheduled[0].callback();
  assert.equal(autoUpdater.checks, 1);
});

test("a repository with no release yet is not an error anyone sees", async () => {
  const { autoUpdater, states, updater } = started();
  const noReleases = new Error("No published versions on GitHub");
  autoUpdater.checkResult = () => {
    autoUpdater.emit("error", noReleases);
    return Promise.reject(noReleases);
  };
  updater.check();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(updater.state().phase, "idle");
  assert.deepEqual(states, []);
});

test("a check that throws at once is contained too", async () => {
  const { autoUpdater, updater } = started();
  autoUpdater.checkResult = () => {
    throw new Error("no app-update.yml");
  };
  updater.check();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(updater.state().phase, "idle");
});

test("an update is downloaded in the background and waits to be installed", () => {
  const { autoUpdater, states, updater } = started();
  autoUpdater.emit("update-available", { version: "1.5.0" });
  assert.deepEqual(updater.state(), { phase: "downloading", version: "1.5.0", checked: true });
  assert.equal(autoUpdater.downloads, 1);
  // Nothing looks again while the download runs: that would start a second one.
  updater.check();
  assert.equal(autoUpdater.checks, 0);
  autoUpdater.emit("update-downloaded", { version: "1.5.0" });
  assert.equal(updater.state().phase, "ready");
  assert.deepEqual(states.map((state) => state.phase), ["downloading", "ready"]);
  assert.deepEqual(autoUpdater.installs, []);
});

test("a download that fails does not become an unhandled rejection", async (t) => {
  const unhandled = [];
  const listener = (reason) => unhandled.push(reason);
  process.on("unhandledRejection", listener);
  t.after(() => process.off("unhandledRejection", listener));
  const { autoUpdater, updater } = started();
  autoUpdater.downloadResult = () => {
    const failure = new Error("net::ERR_CONNECTION_RESET");
    autoUpdater.emit("error", failure);
    return Promise.reject(failure);
  };
  autoUpdater.emit("update-available", { version: "1.5.0" });
  await tick();
  await tick();
  assert.deepEqual(unhandled, []);
  assert.equal(updater.state().phase, "idle");
});

test("a download that cannot even start is contained too", () => {
  const { autoUpdater, updater } = started();
  autoUpdater.downloadResult = () => {
    throw new Error("Please check update first");
  };
  autoUpdater.emit("update-available", { version: "1.5.0" });
  assert.equal(updater.state().phase, "downloading");
  autoUpdater.emit("error", new Error("Please check update first"));
  assert.equal(updater.state().phase, "idle");
});

test("a download that fails is forgotten and tried again at the next check", () => {
  const { autoUpdater, updater } = started();
  autoUpdater.emit("update-available", { version: "1.5.0" });
  autoUpdater.emit("error", new Error("sha512 checksum mismatch"));
  assert.equal(updater.state().phase, "idle");
  updater.check();
  assert.equal(autoUpdater.checks, 1);
});

test("a downloaded update is not thrown away by a later check that fails", () => {
  const { autoUpdater, states, updater } = ready();
  autoUpdater.emit("error", new Error("net::ERR_INTERNET_DISCONNECTED"));
  assert.equal(updater.state().phase, "ready");
  assert.equal(states.length, 2);
});

test("the app keeps looking while a version waits, and the same answer changes nothing", () => {
  const { autoUpdater, states, updater } = ready();
  updater.check();
  assert.equal(autoUpdater.checks, 1);
  autoUpdater.emit("update-available", { version: "1.5.0" });
  autoUpdater.emit("update-downloaded", { version: "1.5.0", downloadedFile: "/cache/pending/again.exe" });
  assert.equal(updater.state().phase, "ready");
  // Not downloaded a second time, and not announced a second time.
  assert.equal(autoUpdater.downloads, 1);
  assert.equal(states.length, 2);
});

test("a downloaded version that is withdrawn is no longer offered", () => {
  const { autoUpdater, updater } = ready();
  // 1.5.0 was taken back: the latest release is the running 1.4.0 again.
  autoUpdater.emit("update-not-available", { version: "1.4.0" });
  assert.deepEqual(updater.state(), { phase: "idle", version: null, checked: true });
  assert.equal(updater.install(), false);
  assert.deepEqual(autoUpdater.installs, []);
});

test("a newer release replaces the one that was waiting", () => {
  const { autoUpdater, updater } = ready();
  autoUpdater.emit("update-available", { version: "1.5.1" });
  assert.deepEqual(updater.state(), { phase: "downloading", version: "1.5.1", checked: true });
  assert.equal(autoUpdater.downloads, 2);
  assert.equal(updater.install(), false);
  autoUpdater.emit("update-downloaded", { version: "1.5.1" });
  assert.deepEqual(updater.state(), { phase: "ready", version: "1.5.1", checked: true });
});

test("restart to update first asks whether the version is still the latest", async () => {
  const same = ready();
  same.autoUpdater.checkResult = () => {
    same.autoUpdater.emit("update-available", { version: "1.5.0" });
    return Promise.resolve({});
  };
  assert.equal(await same.updater.stillReady(), true);
  assert.equal(same.autoUpdater.checks, 1);

  const withdrawn = ready();
  withdrawn.autoUpdater.checkResult = () => {
    withdrawn.autoUpdater.emit("update-not-available", { version: "1.4.0" });
    return Promise.resolve({});
  };
  assert.equal(await withdrawn.updater.stillReady(), false);
  assert.equal(withdrawn.updater.state().phase, "idle");

  const replaced = ready();
  replaced.autoUpdater.checkResult = () => {
    replaced.autoUpdater.emit("update-available", { version: "1.5.1" });
    return Promise.resolve({});
  };
  assert.equal(await replaced.updater.stillReady(), false);
  assert.equal(replaced.updater.state().phase, "downloading");

  assert.equal(await started().updater.stillReady(), false);
});

test("without a network the downloaded version can still be installed", async () => {
  const offline = ready();
  offline.autoUpdater.checkResult = () => {
    const failure = new Error("net::ERR_INTERNET_DISCONNECTED");
    offline.autoUpdater.emit("error", failure);
    return Promise.reject(failure);
  };
  assert.equal(await offline.updater.stillReady(), true);

  // A check that never answers is not waited for beyond RECHECK_MS.
  const silent = ready();
  silent.autoUpdater.checkResult = () => new Promise(() => {});
  const answer = silent.updater.stillReady();
  const patience = silent.timers.scheduled.at(-1);
  assert.equal(patience.delay, RECHECK_MS);
  patience.callback();
  assert.equal(await answer, true);
});

test("a download that has gone is fetched again instead of being installed", async () => {
  const there = new Set(["/cache/pending/AutoGPT-Setup-1.5.0-x64.exe"]);
  const options = { needsDownloadedFile: true, fileExists: (file) => there.has(file) };
  const { autoUpdater, updater } = ready(options);
  assert.equal(await updater.stillReady(), true);
  there.clear();
  const checksBefore = autoUpdater.checks;
  // The runtime has been stopped by now; nothing may be started that quits the app.
  assert.equal(updater.install(), false);
  assert.deepEqual(autoUpdater.installs, []);
  assert.deepEqual(updater.state(), { phase: "idle", version: null, checked: false });
  assert.equal(autoUpdater.checks, checksBefore + 1);

  const asked = ready(options);
  assert.equal(await asked.updater.stillReady(), false);
  assert.equal(asked.updater.state().phase, "idle");
});

test("only Windows and Linux install from the downloaded file", () => {
  assert.equal(installsFromDownloadedFile("win32"), true);
  assert.equal(installsFromDownloadedFile("linux"), true);
  // Squirrel.Mac has its own copy; the archive may be cleaned away.
  assert.equal(installsFromDownloadedFile("darwin"), false);
  const { autoUpdater, updater } = ready({ needsDownloadedFile: false, fileExists: () => false });
  assert.equal(updater.install(), true);
  assert.equal(autoUpdater.installs.length, 1);
});

test("install does nothing until an update has been downloaded", () => {
  const { autoUpdater, updater } = started();
  assert.equal(updater.install(), false);
  autoUpdater.emit("update-available", { version: "1.5.0" });
  assert.equal(updater.install(), false);
  assert.deepEqual(autoUpdater.installs, []);
});

test("installing shows the installer and starts the new version afterwards", () => {
  const { autoUpdater, updater } = started();
  autoUpdater.emit("update-available", { version: "1.5.0" });
  autoUpdater.emit("update-downloaded", { version: "1.5.0" });
  assert.equal(updater.install(), true);
  assert.deepEqual(autoUpdater.installs, [[false, true]]);
  assert.equal(updater.state().phase, "installing");
  assert.equal(updater.install(), false);
  assert.equal(autoUpdater.installs.length, 1);
});

test("an installer that cannot be started is reported, so the app can carry on", () => {
  for (const fails of ["event", "throw"]) {
    const { autoUpdater, states, updater } = started();
    autoUpdater.emit("update-available", { version: "1.5.0" });
    autoUpdater.emit("update-downloaded", { version: "1.5.0" });
    autoUpdater.quitAndInstall = () => {
      if (fails === "throw") throw new Error("spawn EACCES");
      autoUpdater.emit("error", new Error("No update filepath provided"));
    };
    updater.install();
    assert.deepEqual(states.at(-1), { phase: "failed", version: "1.5.0", checked: true });
  }
});

test("an installer that fails after electron-updater has asked the app to quit is reported too", async () => {
  // What NsisUpdater does: quitAndInstall returns, the spawn fails on the
  // next tick, and app.quit() is already scheduled behind it.
  const { autoUpdater, states, updater } = ready();
  const order = [];
  autoUpdater.quitAndInstall = () => {
    process.nextTick(() => autoUpdater.emit("error", new Error("spawn EPERM")));
    setImmediate(() => order.push(`quit while ${updater.state().phase}`));
  };
  assert.equal(updater.install(), true);
  assert.equal(updater.state().phase, "installing");
  await tick();
  assert.deepEqual(states.at(-1), { phase: "failed", version: "1.5.0", checked: true });
  // main.js has started the runtime again by the time the quit arrives.
  assert.deepEqual(order, ["quit while failed"]);
});

test("the quit electron-updater scheduled is dropped once the install has failed", () => {
  assert.equal(dropsUpdaterQuit({ quitIsForUpdate: true, installing: false }), true);
  // The installer is running: this quit is what lets it replace the app.
  assert.equal(dropsUpdaterQuit({ quitIsForUpdate: true, installing: true }), false);
  // The user's own Quit is never dropped.
  assert.equal(dropsUpdaterQuit({ quitIsForUpdate: false, installing: false }), false);
  assert.equal(dropsUpdaterQuit({ quitIsForUpdate: false, installing: true }), false);
});

test("after a failed install nothing is tried again until the app restarts", () => {
  const { autoUpdater, states, updater } = ready();
  autoUpdater.quitAndInstall = () => autoUpdater.emit("error", new Error("No update filepath provided"));
  updater.install();
  const failed = states.at(-1);
  // electron-updater ignores a second quitAndInstall, which would leave the
  // app stopped and waiting for an installer that never comes.
  updater.check();
  assert.equal(autoUpdater.checks, 0);
  for (const type of ["available", "downloaded", "current", "error", "install", "lost"]) {
    assert.equal(nextState(failed, { type, version: "1.5.1" }, "install"), failed);
  }
  assert.equal(updater.install(), false);
});

test("where the app cannot install, a new version is announced once", () => {
  const { autoUpdater, states } = started("notify");
  autoUpdater.emit("update-available", { version: "1.5.0" });
  autoUpdater.emit("update-available", { version: "1.5.0" });
  assert.deepEqual(states, [{ phase: "available", version: "1.5.0", checked: true }]);
  autoUpdater.emit("update-available", { version: "1.6.0" });
  assert.equal(states.at(-1).version, "1.6.0");
});

test("where the app cannot install, a withdrawn version is no longer announced", () => {
  const { autoUpdater, updater } = started("notify");
  autoUpdater.emit("update-available", { version: "1.5.0" });
  updater.check();
  assert.equal(autoUpdater.checks, 1);
  autoUpdater.emit("error", new Error("net::ERR_INTERNET_DISCONNECTED"));
  assert.equal(updater.state().phase, "available");
  autoUpdater.emit("update-not-available", { version: "1.4.0" });
  assert.deepEqual(updater.state(), { phase: "idle", version: null, checked: true });
});

test("next to a failed start, the startup window offers the way to a newer version", () => {
  assert.equal(failedStartOffer(undefined), null);
  assert.equal(failedStartOffer({ phase: "idle", version: null, checked: true }), null);
  assert.deepEqual(failedStartOffer({ phase: "ready", version: "1.5.0" }), {
    label: "Restart to update to 1.5.0",
    action: "install",
  });
  assert.equal(failedStartOffer({ phase: "downloading", version: "1.5.0" }).action, null);
  assert.equal(failedStartOffer({ phase: "available", version: "1.5.0" }).action, "open");
  assert.equal(failedStartOffer({ phase: "failed", version: "1.5.0" }).action, "open");
  assert.equal(failedStartOffer({ phase: "installing", version: "1.5.0" }), null);
});

test("a check can be made before the runtime was ever ready, without arming the timers", () => {
  // main.js does this when the runtime has exited with a fatal error.
  const { autoUpdater, timers, updater } = started();
  updater.check();
  assert.equal(autoUpdater.checks, 1);
  assert.equal(timers.scheduled.length, 0);
});

test("while installing, nothing but a failure changes the state", () => {
  const installing = { phase: "installing", version: "1.5.0", checked: true };
  for (const type of ["available", "downloaded", "current", "install"]) {
    assert.equal(nextState(installing, { type, version: "1.6.0" }, "install"), installing);
  }
  assert.equal(nextState(installing, { type: "error" }, "install").phase, "failed");
});

test("the menu offers what the state allows", () => {
  const calls = [];
  const handlers = {
    check: () => calls.push("check"),
    install: () => calls.push("install"),
    openRelease: (version) => calls.push(`open ${version}`),
  };
  const items = (state) => updateMenuItems({ state, ...handlers });

  assert.deepEqual(items(undefined), []);
  const [check] = items({ phase: "idle", version: null, checked: false });
  assert.equal(check.label, "Check for updates");
  check.click();
  assert.match(items({ phase: "idle", version: null, checked: true })[0].label, /up to date/);
  assert.equal(items({ phase: "downloading", version: "1.5.0" })[0].enabled, false);
  assert.equal(items({ phase: "installing", version: "1.5.0" })[0].enabled, false);
  const [restart] = items({ phase: "ready", version: "1.5.0" });
  assert.equal(restart.label, "Restart to update to 1.5.0");
  restart.click();
  items({ phase: "available", version: "1.5.0" })[0].click();
  items({ phase: "failed", version: "1.5.0" })[0].click();
  assert.deepEqual(calls, ["check", "install", "open 1.5.0", "open 1.5.0"]);
});

test("the Updates menu sits before Window on macOS and last elsewhere", () => {
  const items = [{ label: "Check for updates" }];
  const mac = [{ role: "appMenu" }, { label: "Account" }, { role: "windowMenu" }];
  assert.deepEqual(
    withUpdatesMenu(mac, items, "darwin").map((menu) => menu.role || menu.label),
    ["appMenu", "Account", "Updates", "windowMenu"],
  );
  const other = [{ label: "&Account" }];
  assert.deepEqual(
    withUpdatesMenu(other, items, "win32").map((menu) => menu.label),
    ["&Account", "&Updates"],
  );
  assert.equal(withUpdatesMenu(other, [], "win32"), other);
});

test("electron-updater still has the settings and events this relies on", (t) => {
  const installed = path.join(__dirname, "..", "node_modules", "electron-updater", "out");
  if (!fs.existsSync(installed)) return t.skip("electron-updater is not installed (npm ci)");
  const source = fs.readFileSync(path.join(installed, "AppUpdater.js"), "utf8");
  for (const name of [
    "autoDownload",
    "autoInstallOnAppQuit",
    "autoRunAppAfterInstall",
    "allowPrerelease",
    "allowDowngrade",
    "disableWebInstaller",
  ]) {
    assert.match(source, new RegExp(`this\\.${name} = `), `electron-updater no longer has ${name}`);
  }
  const types = fs.readFileSync(path.join(installed, "types.js"), "utf8");
  for (const event of ["update-available", "update-not-available", "update-downloaded"]) {
    assert.ok(`${source}${types}`.includes(`"${event}"`), `electron-updater no longer emits ${event}`);
  }
  const base = fs.readFileSync(path.join(installed, "BaseUpdater.js"), "utf8");
  assert.ok(base.includes("quitAndInstall(isSilent = false, isForceRunAfter = false)"));
  // What main.js recognises electron-updater's own quit by.
  assert.ok(base.includes('autoUpdater.emit("before-quit-for-update")'), "the quit is no longer announced");
  assert.match(source, /\n {4}downloadUpdate\(/, "electron-updater no longer has downloadUpdate");
  assert.ok(source.includes("downloadedFile: updateFile"), "update-downloaded no longer names the file");
});

test("electron-updater still picks the Linux updater by package-type, and still has the AppImage one", (t) => {
  const main = path.join(__dirname, "..", "node_modules", "electron-updater", "out", "main.js");
  if (!fs.existsSync(main)) return t.skip("electron-updater is not installed (npm ci)");
  const source = fs.readFileSync(main, "utf8");
  assert.ok(source.includes("exports.AppImageUpdater"), "electron-updater no longer exports AppImageUpdater");
  // If this goes, chooseAutoUpdater may no longer be needed; look again.
  assert.ok(source.includes('"package-type"'), "electron-updater no longer reads package-type");
});
