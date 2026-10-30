"use strict";

// When and how the app updates itself, without Electron so it can be tested.
// main.js hands in one of electron-updater's updaters and does what the
// state says.
//
// Releases are GitHub Releases of ntindle/autogpt (the release workflow on
// its `main` branch). There is one channel: the release GitHub marks as the
// latest. electron-updater reads latest*.yml from it, and accepts a download
// only if its SHA-512 is the one in that file; on Windows it also checks the
// publisher once builds are signed, and on macOS Squirrel checks that the
// new app is signed by the same Developer ID as the running one.
//
// Three rules, each of which is a way an update could otherwise lose data:
//
// - Nothing happens while the runtime is starting. Database migrations run
//   then, and an update must never land in one. The app looks for updates
//   once the runtime is ready, or once it has given up and exited.
// - A download never installs itself, not even when the app quits. The
//   installer on Windows takes minutes with nothing on screen, and the user
//   could start the app again underneath it.
// - Installing is "Restart to update": main.js stops the runtime first, which
//   waits out anything that must not be interrupted, and only then calls
//   install().
//
// A version that has been downloaded or announced is not kept for ever: the
// checks go on, and when the latest release is no longer that version (it
// was withdrawn, or a newer one is out) the app follows.
//
// Variants (identity.js) are released in the same repository and never
// cross: the normal app must not become an experiment, and an experiment
// must not become the normal app or another experiment.
//
// - A variant's releases are GitHub pre-releases tagged
//   `desktop-<slug>-v<version>`. GitHub's latest release is never a
//   pre-release, and that one release is all electron-updater's GitHub
//   provider looks at for the normal app (allowPrerelease is off).
// - A variant does not use that provider at all. It lists the repository's
//   releases, takes the highest version among the tags that are exactly its
//   own (newestRelease), and points electron-updater's generic provider at
//   the files of that one release.
// - Whatever a release's metadata names, an update is refused unless every
//   file in it is this app's own installer by name (isOwnUpdateFile).

const fs = require("node:fs");

const { NORMAL, isOwnUpdateFile, releaseTag } = require("./identity");

const RELEASES = "https://github.com/ntindle/autogpt/releases";
const RELEASES_API = "https://api.github.com/repos/ntindle/autogpt/releases";
const RELEASES_PER_PAGE = 100;
// Releases come newest first. A variant that has not been released within
// the repository's last 500 releases is not offered an update.
const MAX_RELEASE_PAGES = 5;
// A version with a pre-release part (1.3.0-rc.1) gets a release page and is
// offered to nobody, for a variant as for the normal app.
const PLAIN_VERSION = /^(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)[.](0|[1-9][0-9]*)$/;
const FIRST_CHECK_MS = 60_000;
const CHECK_EVERY_MS = 6 * 60 * 60_000;
// How long "Restart to update" waits to hear whether the version it holds is
// still the latest. Without an answer (no network) it goes ahead.
const RECHECK_MS = 10_000;
const MAX_LOG_MESSAGE = 2000;
// What a build has when the release workflow did not name its version
// (electron-builder.config.js): 0.0.0-dev.<run>, and 0.0.1-dev.<run> for the
// second build the upgrade test installs over it.
const DEVELOPMENT_VERSION = /^0\.0\.\d+-dev\./;
// Squirrel.Mac replaces the app where it is, which it cannot do on a
// read-only disk image or in the random folder macOS runs a quarantined app
// from until it has been moved.
const MAC_NOT_INSTALLED = /^\/Volumes\/|\/AppTranslocation\//;

// "off"      never looks for updates
// "install"  downloads in the background, installs on "Restart to update"
// "notify"   says that a version is available and links to it
function updateMode({ version, isPackaged, platform, env = {}, macDeveloperId = false, appPath = "" }) {
  if (!isPackaged || DEVELOPMENT_VERSION.test(version)) return "off";
  if (env.AUTOGPT_DESKTOP_UPDATES === "off") return "off";
  if (platform === "win32") return "install";
  if (platform === "darwin") {
    return macDeveloperId && !MAC_NOT_INSTALLED.test(appPath) ? "install" : "notify";
  }
  // An AppImage replaces its own file. A .deb belongs to the package
  // manager and needs root.
  return env.APPIMAGE ? "install" : "notify";
}

// Which of electron-updater's updaters to use. Its `autoUpdater` chooses on
// Linux by the file resources/package-type alone, and electron-builder
// writes that file for the .deb into the directory the AppImage is made from
// as well, at the same time. An AppImage that got it would download the .deb
// and install it with dpkg, next to itself. What says "this is an AppImage"
// is APPIMAGE, which the AppImage's own launcher sets.
function chooseAutoUpdater(electronUpdater, { platform, env = {} }) {
  if (platform === "linux" && env.APPIMAGE) return new electronUpdater.AppImageUpdater();
  return electronUpdater.autoUpdater;
}

// Windows runs the downloaded installer and an AppImage moves the downloaded
// file over itself (after deleting itself), so the file has to be there
// still. On macOS Squirrel has taken its own copy by the time the update is
// ready.
function installsFromDownloadedFile(platform) {
  return platform !== "darwin";
}

function releaseUrl(version, id = NORMAL) {
  if (version) return `${RELEASES}/tag/${releaseTag(id, version)}`;
  // The repository's latest release is the normal app's.
  return id.variant ? RELEASES : `${RELEASES}/latest`;
}

// The release a variant updates to: of the published releases whose tag is
// exactly `desktop-<slug>-v<X.Y.Z>`, the one with the highest version. Null
// when there is none. `releases` is GitHub's list (tag_name, draft).
function newestRelease(releases, id) {
  let newest = null;
  for (const release of releases) {
    const tag = release?.tag_name;
    if (release?.draft || typeof tag !== "string" || !tag.startsWith(id.tagPrefix)) continue;
    const version = tag.slice(id.tagPrefix.length);
    const numbers = PLAIN_VERSION.exec(version)?.slice(1).map(Number);
    if (!numbers) continue;
    if (!newest || isHigher(numbers, newest.numbers)) newest = { tag, version, numbers };
  }
  return newest && { tag: newest.tag, version: newest.version };
}

function isHigher(numbers, than) {
  const differs = numbers.findIndex((number, index) => number !== than[index]);
  return differs !== -1 && numbers[differs] > than[differs];
}

// Where electron-updater's generic provider finds latest*.yml and the
// installers of one release.
function feedUrl(tag) {
  return `${RELEASES}/download/${tag}`;
}

// The feed of one release, as electron-updater's generic provider takes it:
// in app-update.yml (electron-builder.config.js) and when a variant moves on
// to a newer release. GitHub answers a request for several ranges at once
// with 501, and the generic provider would send one for every differential
// download and then fetch the whole installer instead; its GitHub provider
// turns them off for the same reason.
function releaseFeed(tag) {
  return { provider: "generic", url: feedUrl(tag), useMultipleRangeRequest: false };
}

// For a variant: asks GitHub for the repository's releases and picks this
// variant's newest. `fetchJson` resolves to the parsed answer of a URL and
// rejects on anything else.
function variantReleaseFinder({ id, fetchJson }) {
  return async function findRelease() {
    const releases = [];
    for (let page = 1; page <= MAX_RELEASE_PAGES; page += 1) {
      const listed = await fetchJson(`${RELEASES_API}?per_page=${RELEASES_PER_PAGE}&page=${page}`);
      if (!Array.isArray(listed)) throw new Error("GitHub did not answer with a list of releases");
      releases.push(...listed);
      if (listed.length < RELEASES_PER_PAGE) break;
    }
    return newestRelease(releases, id);
  };
}

const IDLE = { phase: "idle", version: null, checked: false };
const UP_TO_DATE = { phase: "idle", version: null, checked: true };
const HOLDS_A_VERSION = ["available", "downloading", "ready"];

// phase: idle | available | downloading | ready | installing | failed
//
// Events: `available`, `downloaded` and `current` are answers from
// electron-updater; `error` is a failed check, download or install; `install`
// is "Restart to update"; `lost` is the downloaded file having gone.
function nextState(state, event, mode) {
  if (state.phase === "installing") {
    return event.type === "error" ? { ...state, phase: "failed" } : state;
  }
  // electron-updater starts an installer once per run of the app, so there
  // is no second attempt: the menu links to the download until the restart.
  if (state.phase === "failed") return state;
  if (event.type === "available") {
    // The same answer as last time is not news.
    if (HOLDS_A_VERSION.includes(state.phase) && state.version === event.version) return state;
    return { phase: mode === "install" ? "downloading" : "available", version: event.version, checked: true };
  }
  if (event.type === "downloaded") {
    if (state.phase === "ready" && state.version === event.version) return state;
    return { phase: "ready", version: event.version, checked: true };
  }
  if (event.type === "install") return state.phase === "ready" ? { ...state, phase: "installing" } : state;
  // The latest release is what this app already is: a version that was
  // downloaded or announced before has been withdrawn.
  if (event.type === "current") return state.phase === "idle" && state.checked ? state : UP_TO_DATE;
  if (event.type === "lost") return state.phase === "ready" ? IDLE : state;
  if (event.type === "error") {
    // A check that fails (no network) says nothing about a version that is
    // already downloaded or announced. A failed download is tried again at
    // the next check. No release at all (a new repository) is one of these.
    if (state.phase === "ready" || state.phase === "available") return state;
    return state.phase === "idle" && !state.checked ? state : IDLE;
  }
  return state;
}

function configure(autoUpdater, logger) {
  autoUpdater.logger = logger;
  // Downloads are started from here (createUpdater), so that looking again
  // while a version waits does not fetch it a second time.
  autoUpdater.autoDownload = false;
  autoUpdater.autoInstallOnAppQuit = false;
  autoUpdater.autoRunAppAfterInstall = true;
  // electron-updater turns this on by itself for a version such as
  // 1.2.3-rc.1, and then looks for tags that are plain versions, which
  // `desktop-v1.2.3` is not: it would find nothing, for ever.
  autoUpdater.allowPrerelease = false;
  autoUpdater.allowDowngrade = false;
  autoUpdater.disableWebInstaller = true;
  autoUpdater.fullChangelog = false;
}

function createUpdater({
  autoUpdater,
  mode,
  logger = null,
  onChange = () => {},
  timers = globalThis,
  needsDownloadedFile = false,
  fileExists = fs.existsSync,
  identity: id = NORMAL,
  findRelease = null,
}) {
  if (id.variant && !findRelease) {
    // Without it the only thing left to ask would be the normal app's feed.
    throw new Error("a variant needs findRelease to look for its own releases");
  }
  let state = IDLE;
  let armed = false;
  let downloadedFile = null;
  configure(autoUpdater, logger);

  function apply(event) {
    const next = nextState(state, event, mode);
    if (next === state) return false;
    state = next;
    onChange(state);
    return true;
  }

  // Every file the release's metadata names must be this app's installer.
  // A release of another variant, or of the normal app, that ended up where
  // this app looks is refused here, before anything is downloaded.
  function isOwnUpdate(info) {
    const named = [...(info.files || []).map((file) => file?.url), ...(info.path ? [info.path] : [])];
    return named.every((name) => isOwnUpdateFile(id, info.version, name));
  }

  autoUpdater.on("update-available", (info) => {
    if (!isOwnUpdate(info)) {
      logger?.error(`Refused version ${info.version}: its files are not ${id.productName}'s.`);
      apply({ type: "error" });
      return;
    }
    const changed = apply({ type: "available", version: info.version });
    if (changed && state.phase === "downloading") download();
  });
  autoUpdater.on("update-downloaded", (info) => {
    downloadedFile = info.downloadedFile || null;
    apply({ type: "downloaded", version: info.version });
  });
  autoUpdater.on("update-not-available", () => apply({ type: "current" }));
  autoUpdater.on("error", () => apply({ type: "error" }));

  // The 'error' event has already said what went wrong; the rejection of the
  // same failure must not become an unhandled one. Resolves when the answer
  // (or the failure) has been applied.
  function settled(start, what) {
    try {
      return Promise.resolve(start()).then(
        () => {},
        () => {},
      );
    } catch (error) {
      logger?.error(`Could not ${what}: ${error.stack || error}`);
      return Promise.resolve();
    }
  }

  function check() {
    if (["downloading", "installing", "failed"].includes(state.phase)) return Promise.resolve();
    return settled(id.variant ? checkOwnReleases : () => autoUpdater.checkForUpdates(), "check for updates");
  }

  // A variant: find its newest release, then let electron-updater read that
  // release and nothing else. No release (none yet, or all withdrawn) is
  // "nothing newer"; a list that could not be read is a failed check.
  async function checkOwnReleases() {
    let release;
    try {
      release = await findRelease();
      if (release) autoUpdater.setFeedURL(releaseFeed(release.tag));
    } catch (error) {
      logger?.warn(`Could not find this variant's releases: ${error.message || error}`);
      apply({ type: "error" });
      return;
    }
    if (!release) {
      apply({ type: "current" });
      return;
    }
    await autoUpdater.checkForUpdates();
  }

  function download() {
    return settled(() => autoUpdater.downloadUpdate(), "download the update");
  }

  function downloadIsThere() {
    return !needsDownloadedFile || !downloadedFile || fileExists(downloadedFile);
  }

  // Something removed the downloaded file (a cleaner, antivirus): forget the
  // version and fetch it again.
  function forgetLostDownload() {
    logger?.warn(`The downloaded update is gone (${downloadedFile}); downloading it again.`);
    downloadedFile = null;
    apply({ type: "lost" });
    check();
  }

  return {
    state: () => state,
    check,
    // Call when the runtime is ready; calling it again changes nothing.
    arm() {
      if (armed) return;
      armed = true;
      timers.setTimeout(check, FIRST_CHECK_MS).unref?.();
      timers.setInterval(check, CHECK_EVERY_MS).unref?.();
    },
    // Before asking the user to restart: is the downloaded version still the
    // latest release, and is its file still there? Looks once more. False
    // when not; the state has moved on by then, and the menus with it.
    async stillReady() {
      if (state.phase !== "ready") return false;
      const { version } = state;
      let timer;
      const patience = new Promise((resolve) => {
        timer = timers.setTimeout(resolve, RECHECK_MS);
      });
      await Promise.race([check(), patience]);
      timers.clearTimeout?.(timer);
      if (state.phase !== "ready" || state.version !== version) return false;
      if (downloadIsThere()) return true;
      forgetLostDownload();
      return false;
    },
    // Only after the runtime has stopped. Returns false when nothing was
    // started and the app should carry on; a failure of the installer
    // arrives as the phase "failed", possibly after this has returned.
    install() {
      if (state.phase !== "ready") return false;
      if (!downloadIsThere()) {
        forgetLostDownload();
        return false;
      }
      apply({ type: "install" });
      try {
        // Not silent: the Windows installer shows its progress window, and
        // every platform starts the new version afterwards.
        autoUpdater.quitAndInstall(false, true);
      } catch (error) {
        logger?.error(`Could not start the installer: ${error.stack || error}`);
        apply({ type: "error" });
      }
      return true;
    },
  };
}

// electron-updater quits the app right after it has started the installer,
// and has no way to take that back when the installer then fails to start.
// By the time that quit arrives the app has heard of the failure and is
// starting its runtime again (installing: false), so the quit is dropped.
function dropsUpdaterQuit({ quitIsForUpdate, installing }) {
  return quitIsForUpdate && !installing;
}

// The entries for the tray menu and the application menu.
function updateMenuItems({ state, check, install, openRelease }) {
  if (!state) return [];
  const { phase, version } = state;
  if (phase === "ready") return [{ label: `Restart to update to ${version}`, click: install }];
  if (phase === "downloading") return [{ label: `Downloading version ${version}…`, enabled: false }];
  if (phase === "installing") return [{ label: `Installing version ${version}…`, enabled: false }];
  if (phase === "available") {
    return [{ label: `Version ${version} is available…`, click: () => openRelease(version) }];
  }
  if (phase === "failed") {
    return [{ label: `Update failed. Download version ${version}…`, click: () => openRelease(version) }];
  }
  return [{ label: state.checked ? "AutoGPT is up to date. Check again" : "Check for updates", click: check }];
}

// What the startup window offers next to an error that stopped AutoGPT from
// starting: the way to a version that may have fixed it. Null when there is
// nothing to offer.
function failedStartOffer(state) {
  if (!state) return null;
  const { phase, version } = state;
  if (phase === "ready") return { label: `Restart to update to ${version}`, action: "install" };
  if (phase === "downloading") return { label: `Downloading version ${version}…`, action: null };
  if (phase === "available" || phase === "failed") return { label: `Download version ${version}`, action: "open" };
  return null;
}

// Adds an Updates menu to owner.js's application menu: before Window on
// macOS, where that one is last by convention, and at the end elsewhere.
function withUpdatesMenu(template, items, platform) {
  if (items.length === 0) return template;
  const menu = { label: platform === "darwin" ? "Updates" : "&Updates", submenu: items };
  if (platform !== "darwin") return [...template, menu];
  return [...template.slice(0, -1), menu, ...template.slice(-1)];
}

// What electron-updater logs (what it found, each download, each failure),
// kept next to the runtime's logs. A failed check quotes GitHub's whole
// answer, tens of kilobytes of it, four times a day.
function fileLogger(file) {
  function write(level, message) {
    const text = message instanceof Error ? message.stack || message.message : String(message);
    const line = text.length > MAX_LOG_MESSAGE ? `${text.slice(0, MAX_LOG_MESSAGE)} [cut]` : text;
    fs.appendFile(file, `${new Date().toISOString()} ${level} ${line}\n`, () => {});
  }
  return {
    info: (message) => write("info", message),
    warn: (message) => write("warn", message),
    error: (message) => write("error", message),
  };
}

module.exports = {
  CHECK_EVERY_MS,
  FIRST_CHECK_MS,
  RECHECK_MS,
  chooseAutoUpdater,
  createUpdater,
  dropsUpdaterQuit,
  failedStartOffer,
  feedUrl,
  releaseFeed,
  fileLogger,
  installsFromDownloadedFile,
  newestRelease,
  nextState,
  releaseUrl,
  updateMenuItems,
  updateMode,
  variantReleaseFinder,
  withUpdatesMenu,
};
