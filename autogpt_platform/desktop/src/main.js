"use strict";

const fs = require("node:fs");
const path = require("node:path");
const {
  app,
  BrowserWindow,
  Menu,
  Notification,
  Tray,
  clipboard,
  dialog,
  ipcMain,
  nativeImage,
  net,
  powerMonitor,
  session,
  shell,
} = require("electron");

const { claudeCodeMenuItems, claudeCodeReports } = require("./claude-code");
const { claim, identity, runtimeEnvironment, variantOf, windowTitle } = require("./identity");
const { allowsPermission, classifyMainNavigation, classifyWindowOpen } = require("./navigation");
const { hearsRuntime } = require("./lifecycle");
const { claimLaunch } = require("./maintenance");
const { applicationMenuTemplate, ownerMenuItems } = require("./owner");
const { claimDataDir, defaultDataDir, readRuntimeManifest, runtimeDir } = require("./paths");
const {
  IDLE,
  activityMenuItems,
  activityOf,
  answerStep,
  closeStep,
  isActive,
  quitAnswer,
  quitQuestion,
  quitStep,
  sessionEndStep,
  sessionIsEnding,
  trayToolTip,
  updateAnswer,
  updateQuestion,
  withQuitMenu,
} = require("./quit");
const { openResetPasswordWindow } = require("./reset-password-window");
const { Runtime } = require("./runtime");
const { collectDoctorReport, exportDoctorReport } = require("./doctor");
const { openDoctorWindow } = require("./doctor-window");
const { applyPreferences, backgroundOnClose, loginAvailability, loginOptions } = require("./desktop-behavior");
const { DEFAULT_PREFERENCES, loadPreferences, savePreferences } = require("./preferences");
const { createPrivacyReporter } = require("./telemetry");
const {
  chooseAutoUpdater,
  createUpdater,
  dropsUpdaterQuit,
  failedStartOffer,
  fileLogger,
  installsFromDownloadedFile,
  releaseUrl,
  updateMenuItems,
  updateMode,
  variantReleaseFinder,
  withUpdatesMenu,
} = require("./updater");

const ICON = path.join(__dirname, "icon.png");
const RELEASE_LIST_TIMEOUT_MS = 30_000;
// Which install this is: the normal app, or a variant that shares nothing
// with it (identity.js). Claimed before the single-instance lock below,
// which Electron keeps in the profile directory.
const install = identity(
  variantOf({ isPackaged: app.isPackaged, manifest: require("../package.json"), env: process.env }),
);
const dataDir = defaultDataDir(process.platform, process.env, install);
let launchRefusal = null;
try {
  claimLaunch(dataDir);
  if (process.platform === "win32") launchRefusal = dataOfAnotherInstall();
} catch (error) {
  launchRefusal = error?.message || "AutoGPT could not open its data folder.";
}
if (!launchRefusal) claim(app, install);
const logsDir = path.join(dataDir, "logs");
const settingsFile = path.join(dataDir, "config", "settings.env");

let runtime = null;
let startupWindow = null;
let mainWindow = null;
let tray = null;
let appUrl = null;
let quitting = false;
let restarting = false;
let failure = null;
let foreignData = null;
// The deployment connection reported at startup (claude-code.js).
const claudeCode = claudeCodeReports(() => refreshMenus());
let updater = null;
let updaterStarted = false;
let confirmingUpdate = false;
let installingUpdate = false;
let quitIsForUpdate = false;
// What is running, as the runtime last counted it, and what the user has
// said about quitting while it runs (quit.js).
let activity = IDLE;
let asking = false;
let stoppingRuns = false;
// When the system said that it is shutting down or that the user is logging
// off: for a while from then (quit.js) nobody is asked anything, and what is
// running resumes on the next start.
let sessionEndingSince = null;
const history = [];
let preferences = DEFAULT_PREFERENCES;
let privacy = null;

if (launchRefusal) {
  dialog.showErrorBox("AutoGPT cannot open its data folder", launchRefusal);
  app.exit(1);
} else if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => focusWindow());
  app.on("web-contents-created", (_, contents) => routeNewWindows(contents));
  app.on("browser-window-created", (_, window) => nameWindow(window));
  app.on("browser-window-created", (_, window) => window.on("session-end", endWithTheSession));
  // macOS: a click on the Dock icon, which is all there is to click while
  // the app runs without a window.
  app.on("activate", () => focusWindow());
  app.whenReady().then(boot);
}

function boot() {
  restrictPermissions(session.defaultSession);
  createTray();
  quitUnaskedWhenToldToEnd();
  if (process.platform !== "win32") foreignData = dataOfAnotherInstall();
  if (!foreignData) {
    fs.mkdirSync(logsDir, { recursive: true });
    preferences = loadPreferences(dataDir);
    privacy = createPrivacyReporter({
      preferences, version: app.getVersion(),
      destinations: require("../package.json").desktop?.telemetry,
    });
    privacy.reportUsage("app_started");
  }
  refreshMenus();
  showStartupWindow();
  if (foreignData) report({ event: "error", fatal: true, message: foreignData });
  else startRuntime();
}

// What to say when the data directory belongs to another install (paths.js
// claimDataDir), or null. Nothing is started on such a directory and
// nothing is written to it: this install's migrations would run on the
// other one's database.
function dataOfAnotherInstall() {
  let owner;
  try {
    owner = claimDataDir(dataDir, install, {
      runtime: runtimeDir({ isPackaged: app.isPackaged, resourcesPath: process.resourcesPath }),
    });
  } catch (error) {
    return `AutoGPT cannot safely open its data folder: ${error?.message || "Data folder preparation failed."}`;
  }
  if (owner === install.dirName) return null;
  return (
    `${dataDir} holds the data of another install of AutoGPT (${owner}), and ${install.productName} ` +
    "will not open it. If AUTOGPT_DESKTOP_DATA_DIR is set, it is what led here: change or remove it, then start again."
  );
}

function startRuntime() {
  let manifest;
  const dir = runtimeDir({ isPackaged: app.isPackaged, resourcesPath: process.resourcesPath });
  try {
    manifest = readRuntimeManifest(dir);
  } catch (error) {
    try {
      fs.appendFileSync(path.join(logsDir, "runtime.log"), `${new Date().toISOString()} Runtime could not be loaded: ${error.stack || error.message}\n`);
    } catch (logError) {
      console.error("Could not write the runtime failure to the log:", logError);
    }
    report({
      event: "error",
      fatal: true,
      message: "AutoGPT's bundled runtime is missing or incomplete. Reinstall AutoGPT from a complete installer. " +
        "Your saved accounts and agents stay in the data folder. Show logs for the missing file details.",
    });
    lookForAFixedVersion();
    return;
  }

  const started = new Runtime({
    command: manifest.command,
    args: manifest.args,
    cwd: dir,
    env: {
      ...manifest.env,
      ...runtimeEnvironment(install),
      AUTOGPT_DESKTOP_DATA_DIR: dataDir,
      AUTOGPT_DESKTOP_SHELL_VERSION: app.getVersion(),
    },
    logFile: path.join(logsDir, "runtime.log"),
    registryFile: path.join(dataDir, "run", "children.json"),
  });
  runtime = started;
  activity = IDLE;
  claudeCode.runtimeStarting();
  // See lifecycle.js: a runtime that is being replaced is not listened to,
  // and neither is one that is being stopped to install an update.
  const heard = (event) =>
    hearsRuntime({ current: runtime, sender: started, restarting: restarting || installingUpdate, event });
  started.on("event", (event) => heard(event) && onRuntimeEvent(event));
  started.on("exit", ({ code, expected }) => {
    // Nothing runs without it: a quit from here on has nothing to ask about.
    if (runtime === started) showActivity(IDLE);
    if (runtime !== started) return;
    if (expected || quitting) {
      privacy?.reportUsage("runtime_stopped");
      return;
    }
    privacy?.reportCrash("runtime_exit");
    // A runtime that failed has usually said why already; keep its words.
    const message = failure || `AutoGPT stopped unexpectedly (exit code ${code}).`;
    if (!failure) report({ event: "error", fatal: true, message });
    if (mainWindow) explainCrash(message);
    else lookForAFixedVersion();
  });
  started.start();
}

function onRuntimeEvent(event) {
  if (claudeCode.hears(event)) return;
  // Not for `report`: a count is not a step of the start.
  const counted = activityOf(event);
  if (counted) return showActivity(counted);
  if (event.event === "activity") return;
  report(event);
  if (event.event === "ready" && typeof event.url === "string") {
    privacy?.reportUsage("runtime_ready");
    appUrl = event.url;
    openMainWindow();
    // Not before the first `ready`: migrations are over by then.
    theUpdater()?.arm();
    refreshMenus();
  }
}

// AutoGPT could not start and its runtime has exited. If that is this
// version's fault, the fix is a newer version, and the user must not have
// to go and find it: look once, now, and offer it next to the error.
function lookForAFixedVersion() {
  theUpdater()?.check();
  refreshMenus();
}

// See updater.js for when the app looks for updates and what it accepts.
// Null when this build does not update itself, and when electron-updater
// cannot be started: updating is optional, the app is not.
function theUpdater() {
  if (updaterStarted) return updater;
  updaterStarted = true;
  try {
    updater = startUpdater();
  } catch (error) {
    fileLogger(path.join(logsDir, "updater.log")).error(`Updates are off: ${error.stack || error}`);
  }
  return updater;
}

function startUpdater() {
  const mode = updateMode({
    version: app.getVersion(),
    isPackaged: app.isPackaged,
    platform: process.platform,
    env: process.env,
    macDeveloperId: Boolean(require("../package.json").autogptDesktop?.macDeveloperId),
    appPath: app.getAppPath(),
  });
  if (mode === "off") return null;
  // electron-updater says so on Electron's own updater, immediately before
  // it quits the app.
  require("electron").autoUpdater.on("before-quit-for-update", () => {
    quitIsForUpdate = true;
  });
  return createUpdater({
    autoUpdater: chooseAutoUpdater(require("electron-updater"), { platform: process.platform, env: process.env }),
    mode,
    logger: fileLogger(path.join(logsDir, "updater.log")),
    onChange: onUpdateState,
    needsDownloadedFile: installsFromDownloadedFile(process.platform),
    identity: install,
    findRelease: install.variant ? variantReleaseFinder({ id: install, fetchJson }) : null,
  });
}

// Through Chromium's network stack, like electron-updater's own requests,
// so that the system's proxy settings apply.
async function fetchJson(url) {
  const response = await net.fetch(url, {
    headers: { Accept: "application/vnd.github+json" },
    signal: AbortSignal.timeout(RELEASE_LIST_TIMEOUT_MS),
  });
  if (!response.ok) throw new Error(`${url} answered ${response.status}`);
  return response.json();
}

function onUpdateState({ phase, version }) {
  refreshMenus();
  sendUpdateOffer();
  if (phase === "failed" && installingUpdate) {
    return resumeAfterFailedUpdate("The update could not be installed. Starting AutoGPT again…");
  }
  if (phase !== "ready" && phase !== "available") return;
  if (!Notification.isSupported()) return;
  const body =
    phase === "ready"
      ? "Choose Restart to update in the AutoGPT menu when it suits you."
      : "Open the AutoGPT menu to download it.";
  new Notification({ title: `${install.productName} ${version} is available`, body }).show();
}

// Stops the runtime as a quit does, which waits out a database migration,
// and only then lets the installer replace the app. With agents running,
// the question said they would be stopped, and they are, as "Stop and quit"
// stops them.
async function restartToUpdate() {
  if (quitting || restarting || confirmingUpdate || updater?.state().phase !== "ready") return;
  confirmingUpdate = true;
  let stopRuns;
  try {
    const answer = await userWantsTheUpdate();
    if (!answer.restart) return;
    stopRuns = answer.stopRuns;
  } finally {
    confirmingUpdate = false;
  }
  quitting = true;
  installingUpdate = true;
  appUrl = null;
  history.length = 0;
  // On macOS the new version is only unpacked now, which takes a while.
  history.push({ event: "progress", message: "Installing the update…" });
  showProgressInsteadOfTheApp();
  await runtime?.stop({ stopRuns }).catch(() => {});
  // The AppImage starts the new version before this one has gone, and a
  // second instance quits at once while the first holds the lock.
  if (process.platform === "linux") app.releaseSingleInstanceLock();
  if (!updater.install()) resumeAfterFailedUpdate("The update has to be downloaded again. Starting AutoGPT again…");
  refreshMenus();
}

// Asks, but only for a version that is still the latest release and whose
// download is still there (updater.js looks once more). The answer counts
// only if nothing changed while the question was on screen.
async function userWantsTheUpdate() {
  if (!(await updater.stillReady())) return { restart: false };
  const { version } = updater.state();
  // What the question says is running is what a yes agrees to stop.
  const stopRuns = isActive(activity);
  const response = await showQuestion(updateQuestion({ install, version, activity }));
  const restart = updateAnswer({ response, asked: version, ...updater.state(), quitting, restarting });
  return { restart, stopRuns };
}

// On the window the user is looking at, if there is one.
async function showQuestion(question) {
  const parent = [mainWindow, startupWindow].find((window) => window?.isVisible());
  const { response } = await (parent ? dialog.showMessageBox(parent, question) : dialog.showMessageBox(question));
  return response;
}

// The installer was not started, or could not be: the old version is
// intact, so run it. electron-updater may already have asked the app to
// quit; `before-quit` drops that request (dropsUpdaterQuit).
function resumeAfterFailedUpdate(message) {
  installingUpdate = false;
  quitting = false;
  if (process.platform === "linux") app.requestSingleInstanceLock();
  failure = null;
  history.length = 0;
  history.push({ event: "progress", message });
  showProgressInsteadOfTheApp();
  startRuntime();
}

// What the startup window shows next to a failed start (updater.js).
function sendUpdateOffer() {
  startupWindow?.webContents.send("update-offer", failedStartOffer(updater?.state()));
}

function takeUpdateOffer() {
  const offer = failedStartOffer(updater?.state());
  if (offer?.action === "install") restartToUpdate();
  if (offer?.action === "open") shell.openExternal(releaseUrl(updater.state().version, install));
}

function updateItems() {
  return updateMenuItems({
    state: updater?.state(),
    check: () => updater.check(),
    install: restartToUpdate,
    openRelease: (version) => shell.openExternal(releaseUrl(version, install)),
  });
}

function report(event) {
  if (event.event === "error" && event.fatal) {
    failure = event.message;
    if (!appUrl) privacy?.reportCrash("runtime_start_failed");
  }
  history.push(event);
  startupWindow?.webContents.send("runtime-event", event);
  if (event.event === "error" && event.fatal) focusWindow();
}

function showStartupWindow() {
  startupWindow = new BrowserWindow({
    width: 520,
    height: 420,
    resizable: false,
    title: install.productName,
    icon: ICON,
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "startup-preload.js"),
      contextIsolation: true,
      sandbox: true,
    },
  });
  startupWindow.loadFile(path.join(__dirname, "startup.html"));
  startupWindow.once("ready-to-show", () => startupWindow.show());
  startupWindow.webContents.on("did-finish-load", () => {
    for (const event of history) startupWindow?.webContents.send("runtime-event", event);
    sendUpdateOffer();
  });
  startupWindow.on("closed", () => {
    startupWindow = null;
    if (!mainWindow) app.quit();
  });
}

function openMainWindow() {
  if (mainWindow) return focusWindow();
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: install.productName,
    icon: ICON,
    show: false,
    autoHideMenuBar: true,
    webPreferences: { contextIsolation: true, sandbox: true },
  });
  keepMainWindowInApp(mainWindow);
  mainWindow.loadURL(appUrl);
  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
    startupWindow?.destroy();
  });
  // Closing the window quits the app, so with agents running it is the
  // same decision as a quit (quit.js).
  mainWindow.on("close", (event) => {
    if (backgroundOnClose({ ...preferences, quitting, sessionEnding: sessionEnding(), runtimeExited: runtimeExited() })) {
      event.preventDefault();
      mainWindow.hide();
      return;
    }
    const step = closeStep({ activity, quitting, sessionEnding: sessionEnding() });
    if (step === "close") return;
    event.preventDefault();
    chooseAboutRuns();
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
    app.quit();
  });
  mainWindow.webContents.on("render-process-gone", () => privacy?.reportCrash("renderer_gone"));
}

function desktopLoginAvailability() {
  return loginAvailability({
    isPackaged: app.isPackaged, platform: process.platform,
    customProfile: Boolean(process.env.AUTOGPT_DESKTOP_DATA_DIR) || app.commandLine.hasSwitch("user-data-dir"),
  });
}

function desktopState() {
  const login = desktopLoginAvailability();
  if (login.supported) {
    try {
      const actual = app.getLoginItemSettings(process.platform === "win32" ? { path: process.execPath, args: [] } : {});
      login.status = actual.status === "requires-approval" ? "Approve AutoGPT in your system's login items settings." :
        actual.openAtLogin ? "Registered to launch at login." : "Not registered to launch at login.";
    } catch { login.status = "The operating system's login setting could not be checked."; }
  }
  return {
    version: app.getVersion(), preferences, login,
    reporting: privacy?.configuration() || {
      usageStatistics: { configured: false, host: null }, crashReports: { configured: false, host: null },
    },
  };
}

async function saveDesktopPreferences(patch) {
  if (foreignData) return { error: "This installation cannot change another installation's preferences." };
  const current = preferences;
  const withdrawn = {};
  for (const category of ["usageStatistics", "crashReports"]) {
    if (patch && Object.hasOwn(patch, category) && patch[category] === false) withdrawn[category] = false;
  }
  preferences = { ...current, ...withdrawn };
  privacy?.updatePreferences(preferences);
  try {
    const reporting = privacy?.configuration();
    for (const category of ["usageStatistics", "crashReports"]) {
      if (patch?.[category] === true && !reporting?.[category]?.configured) {
        throw new Error("Reporting is disabled in this build.");
      }
    }
    preferences = await applyPreferences({
      current, patch, supported: desktopLoginAvailability().supported,
      save: update => savePreferences(dataDir, { ...preferences, ...update }),
      setLogin: enabled => app.setLoginItemSettings(loginOptions(enabled, process.platform, process.execPath, install.dirName)),
      getLogin: () => app.getLoginItemSettings(process.platform === "win32" ? { path: process.execPath, args: [] } : {}).openAtLogin,
    });
    privacy?.updatePreferences(preferences);
    return { state: desktopState() };
  } catch {
    return { state: desktopState(), error: Object.keys(withdrawn).length ?
      "Could not save your choices. Reporting you turned off stays off for this session only; retry saving before restarting AutoGPT." :
      "Could not save preferences. Check available disk space and system login settings, then try again." };
  }
}

async function doctorConnectionStatus() {
  if (!appUrl || !mainWindow || runtimeExited()) return "unavailable";
  const contents = mainWindow.webContents;
  try {
    if (new URL(contents.getURL()).origin !== new URL(appUrl).origin) return "unknown";
    let timer;
    try { return await Promise.race([contents.executeJavaScript(`(async () => {
      try {
        const response = await fetch('/api/proxy/api/chat/connections', { signal: AbortSignal.timeout(2500) });
        if ([401,403].includes(response.status)) return 'sign-in-required';
        if (!response.ok) return 'unavailable';
        const body = await response.json();
        if (!Array.isArray(body.offers)) return 'unknown';
        return body.offers.some(offer => offer && offer.selectable === true && offer.state === 'ready' && !offer.lock_reason) ? 'connected' : 'not-connected';
      } catch { return 'unavailable'; }
    })()`), new Promise(resolve => { timer = setTimeout(() => resolve("unavailable"), 3500); })]); }
    finally { clearTimeout(timer); }
  } catch { return "unavailable"; }
}

function showDoctor(tab = "doctor") {
  if (foreignData) return focusWindow();
  openDoctorWindow({
    icon: ICON, getState: desktopState, savePreferences: saveDesktopPreferences,
    openLogs: async () => { if (await shell.openPath(logsDir)) throw new Error("Logs could not be opened"); },
    openConnections: async () => {
      if (!appUrl || runtimeExited()) throw new Error("Runtime unavailable");
      focusWindow();
      await mainWindow?.loadURL(`${appUrl}/settings/integrations`);
    },
    runChecks: async () => {
      const state = failure ? "error" : runtimeExited() ? "stopped" : appUrl ? "ready" : "starting";
      const report = await collectDoctorReport({
        dataDir, runtimeDir: runtimeDir({ isPackaged: app.isPackaged, resourcesPath: process.resourcesPath }),
        version: app.getVersion(), platform: process.platform, arch: process.arch, appUrl,
        runtimeState: state, connectionStatus: await doctorConnectionStatus(),
      });
      privacy?.reportUsage("doctor_run");
      return { report, preview: exportDoctorReport(report) };
    },
  }, tab);
}

function doctorMenuItems() {
  return [
    { label: "AutoGPT Doctor…", click: () => showDoctor() },
    { label: "Privacy & diagnostics…", click: () => showDoctor("privacy") },
    { label: "Desktop preferences…", click: () => showDoctor("desktop") },
  ];
}

// Stop the runtime and start it again, showing progress as a first start
// does. Used after a change that the runtime only picks up while starting.
async function restartRuntime() {
  if (restarting || quitting) return;
  restarting = true;
  appUrl = null;
  failure = null;
  // What is running is interrupted by the restart and resumes after it.
  activity = IDLE;
  history.length = 0;
  history.push({ event: "progress", message: "Restarting AutoGPT…" });
  showProgressInsteadOfTheApp();
  refreshMenus();
  try {
    await runtime?.stop();
  } finally {
    restarting = false;
  }
  if (!quitting) startRuntime();
}

function showProgressInsteadOfTheApp() {
  // A reload clears a failure the startup window may be showing; loading
  // replays `history` either way.
  if (startupWindow) startupWindow.webContents.reload();
  else showStartupWindow();
  if (!mainWindow) return;
  mainWindow.removeAllListeners("closed"); // closing it here is not a quit
  mainWindow.destroy();
  mainWindow = null;
}

function resetOwnerPassword() {
  if (foreignData) return focusWindow();
  openResetPasswordWindow({ icon: ICON, dataDir, onWritten: restartRuntime });
}

// A page names its own window, and the platform's pages name themselves the
// same in every install: a variant's windows say which one they are.
function nameWindow(window) {
  window.on("page-title-updated", (event, title) => {
    const named = windowTitle(install, title);
    if (named === null) return;
    event.preventDefault();
    window.setTitle(named);
  });
}

// See navigation.js for what stays in the app and why.
function routeNewWindows(contents) {
  contents.setWindowOpenHandler((details) => {
    const destination = classifyWindowOpen(details, appUrl);
    if (destination === "browser") shell.openExternal(details.url);
    if (destination !== "app") return { action: "deny" };
    return {
      action: "allow",
      overrideBrowserWindowOptions: {
        icon: ICON,
        autoHideMenuBar: true,
        webPreferences: { contextIsolation: true, sandbox: true },
      },
    };
  });
}

function keepMainWindowInApp(window) {
  function route(event, url) {
    const destination = classifyMainNavigation(url, appUrl);
    if (destination === "app") return;
    event.preventDefault();
    if (destination === "browser") shell.openExternal(url);
  }
  window.webContents.on("will-navigate", route);
  // One of the app's own links can answer with a redirect to another site.
  window.webContents.on("will-redirect", route);
}

// See navigation.js for what is allowed and why.
function restrictPermissions(target) {
  target.setPermissionRequestHandler((contents, permission, callback, details) => {
    const origin = details.requestingUrl || contents.getURL();
    callback(allowsPermission({ permission, origin, mediaTypes: details.mediaTypes }, appUrl));
  });
  target.setPermissionCheckHandler((_contents, permission, origin, details) =>
    allowsPermission({ permission, origin, mediaTypes: [details.mediaType] }, appUrl),
  );
}

// The backend is gone, so the page in the window can only fail from here on.
async function explainCrash(message) {
  const { response } = await dialog.showMessageBox(mainWindow, {
    type: "error",
    message,
    detail: "Open Doctor to check this installation, or inspect the local logs.",
    buttons: ["Open Doctor", "Show logs", "Close"],
    defaultId: 0,
    cancelId: 2,
  });
  if (response === 0) return showDoctor();
  if (response === 1) shell.openPath(logsDir);
  app.quit();
}

// The tray icon can be hidden or missing, so its owner actions are in the
// application menu as well (owner.js; outside macOS it shows on Alt), and so
// is Quit (quit.js).
function applicationMenu() {
  const template = applicationMenuTemplate(process.platform, [...ownerItems(), ...autopilotItems()]);
  template.push({ label: "Help", submenu: doctorMenuItems() });
  const withUpdates = withUpdatesMenu(template, updateItems(), process.platform);
  return Menu.buildFromTemplate(withQuitMenu(withUpdates, install, process.platform));
}

function focusWindow() {
  const window = mainWindow || startupWindow;
  if (!window) return;
  if (window.isMinimized()) window.restore();
  window.show();
  window.focus();
}

function createTray() {
  tray = new Tray(nativeImage.createFromPath(ICON).resize({ width: 16, height: 16 }));
  tray.on("click", focusWindow);
}

function trayMenu() {
  return Menu.buildFromTemplate([
    // First: whether anything is running, without opening the window.
    ...activityMenuItems(activity),
    { label: `Open ${install.productName}`, click: focusWindow },
    { label: "Open in browser", click: () => appUrl && shell.openExternal(appUrl) },
    { label: "Settings file (API keys)", click: () => shell.openPath(settingsFile) },
    { label: "Show logs", click: () => shell.openPath(logsDir) },
    ...doctorMenuItems(),
    { type: "separator" },
    ...ownerItems(),
    ...autopilotItems(),
    { type: "separator" },
    ...updateItems(),
    { label: `Quit ${install.productName}`, click: () => app.quit() },
  ]);
}

function ownerItems() {
  return ownerMenuItems({
    appUrl,
    copyText: (text) => clipboard.writeText(text),
    resetPassword: resetOwnerPassword,
  });
}

// Below the owner's entries in both menus, under a separator of its own.
function autopilotItems() {
  const items = claudeCodeMenuItems({ status: claudeCode.status(), install });
  return items.length ? [{ type: "separator" }, ...items] : [];
}

// The menus show the app's address, which is only known once it is ready.
function refreshMenus() {
  Menu.setApplicationMenu(applicationMenu());
  tray?.setContextMenu(trayMenu());
  tray?.setToolTip(trayToolTip(install, activity, stoppingRuns));
}

ipcMain.on("open-logs", () => shell.openPath(logsDir));
ipcMain.on("quit", () => app.quit());
ipcMain.on("take-update-offer", takeUpdateOffer);
ipcMain.on("open-doctor", (event) => {
  if (startupWindow && event.sender === startupWindow.webContents && event.senderFrame === startupWindow.webContents.mainFrame) showDoctor();
});
app.on("will-quit", () => privacy?.dispose());

app.on("window-all-closed", () => {});

app.on("before-quit", (event) => {
  const dropped = dropsUpdaterQuit({ quitIsForUpdate, installing: installingUpdate });
  quitIsForUpdate = false;
  if (dropped) return event.preventDefault();
  const step = quitStep({ activity, quitting, sessionEnding: sessionEnding(), runtimeExited: runtimeExited() });
  if (step === "leave") return;
  event.preventDefault();
  if (step === "ask") chooseAboutRuns();
  // A second quit while the runtime is stopping: it waits for nothing more.
  else if (step === "hurry") runtime.stop().catch(() => {});
  else quit();
});

function showActivity(counted) {
  activity = counted;
  refreshMenus();
}

// Put the question of quit.js, once at a time, and do what was answered.
async function chooseAboutRuns() {
  if (asking) return;
  asking = true;
  let answer;
  try {
    answer = quitAnswer(await showQuestion(quitQuestion({ install, activity, platform: process.platform })));
  } finally {
    asking = false;
  }
  const step = answerStep({ answer, quitting, runtimeExited: runtimeExited() });
  if (step === "background") hideWindows();
  if (step === "stop") quit({ stopRuns: true });
}

function runtimeExited() {
  return !runtime || runtime.hasExited;
}

function sessionEnding() {
  return sessionIsEnding(sessionEndingSince, Date.now());
}

function hideWindows() {
  for (const window of BrowserWindow.getAllWindows()) window.hide();
}

// Stop the runtime, then exit. `stopRuns`: the user chose to end what is
// running, which can take a while; the windows go at once, and the tray icon
// says what the app is still there for.
function quit({ stopRuns = false } = {}) {
  quitting = true;
  startupWindow?.webContents.send("runtime-event", {
    event: "progress",
    message: `Stopping ${install.productName}…`,
  });
  if (stopRuns) {
    stoppingRuns = true;
    hideWindows();
    refreshMenus();
  }
  runtime
    .stop({ stopRuns })
    .catch((error) => dialog.showErrorBox(install.productName, `Could not stop cleanly: ${error.message}`))
    .finally(() => app.exit(0));
}

// Windows tells every window that the session is ending, and does not wait
// long. No `before-quit` comes with it. A stop that is already under way,
// for the runs to end, is told to wait for nothing more: called again,
// `stop` says that.
function endWithTheSession() {
  sessionEndingSince = Date.now();
  if (sessionEndStep({ quitting, runtimeExited: runtimeExited() }) === "nothing") return;
  quitting = true;
  runtime.stop().catch(() => {});
}

// macOS and Linux: the system says it is shutting down (or, on macOS, that
// the user is logging off) and then asks the app to quit like anyone else.
// A signal is how a session manager, or `kill`, asks. Neither is the user
// choosing to quit, so neither is asked about running agents.
function quitUnaskedWhenToldToEnd() {
  if (process.platform === "win32") return;
  powerMonitor.on("shutdown", () => {
    sessionEndingSince = Date.now();
  });
  for (const signal of ["SIGTERM", "SIGHUP", "SIGINT"]) {
    process.on(signal, () => {
      sessionEndingSince = Date.now();
      app.quit();
    });
  }
}
