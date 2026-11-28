"use strict";

// The standard and experimental distributions keep separate profiles,
// installer identities and update feeds (src/identity.js).
// Three things are held in place here: the normal app's identity is what it
// was before there were variants; no two installs share a name, a directory
// or a port; and an update never crosses from one to another.

const assert = require("node:assert/strict");
const { EventEmitter } = require("node:events");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { test } = require("node:test");

const {
  NORMAL,
  VARIANT_VARIABLE,
  artifactNames,
  claim,
  identity,
  isOwnUninstallName,
  isOwnUpdateFile,
  releaseTag,
  runtimeEnvironment,
  variantOf,
  windowTitle,
} = require("../src/identity");
const { INSTALL_STAMP, claimDataDir: claimDataDirWithRuntime, defaultDataDir } = require("../src/paths");
const claimDataDir = (directory, install) => claimDataDirWithRuntime(directory, install, { platform: "linux" });
const {
  createUpdater,
  feedUrl,
  newestRelease,
  releaseFeed,
  releaseUrl,
  variantReleaseFinder,
} = require("../src/updater");

const DESKTOP = path.join(__dirname, "..");
const CONFIG = path.join(DESKTOP, "electron-builder.config.js");
const UPDATER = path.join(DESKTOP, "node_modules", "electron-updater", "out");
const SETTINGS = [
  "AUTOGPT_DESKTOP_VERSION",
  "AUTOGPT_DESKTOP_OUTPUT",
  "AUTOGPT_DESKTOP_VARIANT",
  "AUTOGPT_DESKTOP_MAC_SIGN",
  "AUTOGPT_DESKTOP_WIN_SIGN",
];
const VARIANTS = ["experimental"];

// The configuration as electron-builder would load it in this environment.
function configured(settings = {}) {
  const saved = Object.fromEntries(SETTINGS.map((name) => [name, process.env[name]]));
  for (const name of SETTINGS) delete process.env[name];
  Object.assign(process.env, settings);
  delete require.cache[require.resolve(CONFIG)];
  try {
    return require(CONFIG);
  } finally {
    delete require.cache[require.resolve(CONFIG)];
    for (const [name, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  }
}

// The files a release of `id` holds, from the patterns the build uses.
function releaseFiles(id, version) {
  const names = artifactNames(id);
  const fill = (pattern, arch, ext) =>
    pattern.replace("${version}", version).replace("${arch}", arch).replace("${ext}", ext);
  return [
    fill(names.nsis, "x64", "exe"),
    fill(names.mac, "arm64", "dmg"),
    fill(names.mac, "arm64", "zip"),
    fill(names.appImage, "x86_64", "AppImage"),
    fill(names.deb, "amd64", "deb"),
  ];
}

// --- the normal app ---------------------------------------------------------

test("only standard and experimental distributions can be built", () => {
  for (const variant of ["voice", "lab", "feed-proof", "nightly", "updater", "Experimental", "experimental-2", "../x", "x".repeat(25), null, 7, {}]) {
    assert.throws(() => identity(variant), /AUTOGPT_DESKTOP_VARIANT must be/, String(variant));
  }
  assert.equal(identity("").dirName, "AutoGPT");
  assert.equal(identity("experimental").dirName, "AutoGPT-experimental");
});

test("both distribution feeds belong to the upstream organization", () => {
  assert.deepEqual(configured().publish, [{ provider: "github", owner: "Significant-Gravitas", repo: "AutoGPT" }]);
  assert.equal(
    configured({ AUTOGPT_DESKTOP_VARIANT: "experimental", AUTOGPT_DESKTOP_VERSION: "1.4.0" }).publish[0].url,
    "https://github.com/Significant-Gravitas/AutoGPT/releases/download/desktop-experimental-v1.4.0",
  );
});

test("the normal app is exactly what it was before there were variants", () => {
  // Changing any of these orphans every installed app: its data, its
  // sign-ins, its place in the system and its updates.
  assert.deepEqual(identity(), {
    variant: "",
    productName: "AutoGPT",
    appId: "co.agpt.autogpt.desktop",
    packageName: "autogpt",
    executableName: null,
    productFilename: "AutoGPT",
    dirName: "AutoGPT",
    artifactBase: "AutoGPT",
    publicPort: 18473,
    tagPrefix: "desktop-v",
  });
  assert.equal(identity(""), NORMAL);
  assert.deepEqual(artifactNames(NORMAL), {
    nsis: "AutoGPT-Setup-${version}-${arch}.${ext}",
    mac: "AutoGPT-${version}-${arch}.${ext}",
    appImage: "AutoGPT-${arch}.${ext}",
    deb: "AutoGPT-${version}-${arch}.${ext}",
  });
  assert.equal(releaseTag(NORMAL, "1.4.0"), "desktop-v1.4.0");
  assert.throws(() => {
    NORMAL.dirName = "elsewhere";
  }, TypeError);
});

test("the normal app's package is the one in package.json", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(DESKTOP, "package.json"), "utf8"));
  assert.equal(manifest.name, NORMAL.packageName);
  assert.equal(manifest.productName, NORMAL.productName);
  assert.equal(manifest.autogptDesktop, undefined, "a variant is named when building, not in the repository");
});

test("without the variable, and with it empty, the build is the normal app's", () => {
  for (const settings of [{}, { AUTOGPT_DESKTOP_VARIANT: "" }]) {
    const config = configured(settings);
    assert.equal(config.appId, "co.agpt.autogpt.desktop");
    assert.equal(config.productName, "AutoGPT");
    assert.ok(!("executableName" in config), "the normal app's executable is named by electron-builder");
    assert.deepEqual(config.extraMetadata, { autogptDesktop: { macDeveloperId: false } });
    assert.deepEqual(config.publish, [{ provider: "github", owner: "Significant-Gravitas", repo: "AutoGPT" }]);
    assert.equal(config.nsis.artifactName, "AutoGPT-Setup-${version}-${arch}.${ext}");
    assert.equal(config.mac.artifactName, "AutoGPT-${version}-${arch}.${ext}");
    assert.equal(config.appImage.artifactName, "AutoGPT-${arch}.${ext}");
    assert.equal(config.deb.artifactName, "AutoGPT-${version}-${arch}.${ext}");
    assert.deepEqual(Object.keys(config).sort(), [
      "afterPack",
      "appId",
      "appImage",
      "asar",
      "beforePack",
      "deb",
      "detectUpdateChannel",
      "directories",
      "extraMetadata",
      "extraResources",
      "files",
      "forceCodeSigning",
      "linux",
      "mac",
      "npmRebuild",
      "nsis",
      "productName",
      "publish",
      "win",
    ]);
  }
  const versioned = configured({ AUTOGPT_DESKTOP_VERSION: "1.4.0" });
  assert.deepEqual(versioned.extraMetadata, { version: "1.4.0", autogptDesktop: { macDeveloperId: false } });
});

test("the normal app keeps its data, its profile and its name", () => {
  assert.equal(
    defaultDataDir("win32", { LOCALAPPDATA: "C:\\Users\\u\\AppData\\Local" }),
    path.join("C:\\Users\\u\\AppData\\Local", "AutoGPT"),
  );
  assert.equal(defaultDataDir("linux", { XDG_DATA_HOME: "/x/share" }), path.join("/x/share", "AutoGPT"));
  assert.match(defaultDataDir("darwin", {}), /Library[\\/]Application Support[\\/]AutoGPT$/);
  const app = fakeApp();
  claim(app, NORMAL);
  // What Electron picks by itself for a product named AutoGPT.
  assert.deepEqual(app.calls, [
    ["setName", "AutoGPT"],
    ["setPath", "userData", path.join("/appdata", "AutoGPT")],
  ]);
  assert.equal(windowTitle(NORMAL, "AutoGPT Platform"), null);
  assert.equal(windowTitle(NORMAL, ""), null);
});

// --- a variant ----------------------------------------------------------------

test("the experimental identity stays compatible with existing installations", () => {
  assert.deepEqual(identity("experimental"), {
    variant: "experimental",
    productName: "AutoGPT (experimental)",
    appId: "co.agpt.autogpt.desktop.experimental",
    packageName: "autogpt-experimental",
    executableName: "autogpt-experimental",
    productFilename: "autogpt-experimental",
    dirName: "AutoGPT-experimental",
    artifactBase: "AutoGPT-experimental",
    publicPort: 25500,
    tagPrefix: "desktop-experimental-v",
  });
  assert.deepEqual(identity("experimental"), identity("experimental"));
  assert.equal(releaseTag(identity("experimental"), "1.4.0"), "desktop-experimental-v1.4.0");
});

// Every folder an install keeps beside those of other installs: its data and
// profile, and electron-updater's download cache (app-builder-lib appInfo.js
// updaterCacheDirName: the package name, lower-cased, plus "-updater").
function ownedFolders(install) {
  return [install.dirName, `${install.packageName}-updater`].map((name) => name.toLowerCase());
}

test("no folder of one install is a folder of another, on a file system that ignores case", () => {
  const slugs = VARIANTS;
  const installs = [NORMAL, ...slugs.map((slug) => identity(slug))];
  const owners = new Map();
  for (const install of installs) {
    for (const folder of ownedFolders(install)) {
      assert.equal(owners.get(folder), undefined, `${folder} is ${owners.get(folder)}'s and ${install.dirName}'s`);
      owners.set(folder, install.dirName);
    }
  }
  assert.ok(ownedFolders(NORMAL).includes("autogpt-updater"));
  assert.ok(ownedFolders(identity("experimental")).includes("autogpt-experimental-updater"));
});

test("the download cache is named as electron-builder names it", (t) => {
  const appInfo = path.join(DESKTOP, "node_modules", "app-builder-lib", "out", "appInfo.js");
  if (!fs.existsSync(appInfo)) return t.skip("electron-builder is not installed (npm ci)");
  const { AppInfo } = require(appInfo);
  for (const variant of ["", "experimental"]) {
    const config = configured({ AUTOGPT_DESKTOP_VARIANT: variant, AUTOGPT_DESKTOP_VERSION: "1.4.0" });
    const metadata = { ...require("../package.json"), ...config.extraMetadata };
    const info = new AppInfo({ metadata, config, framework: { defaultAppIdPrefix: "x." } }, null, null);
    assert.equal(info.updaterCacheDirName, ownedFolders(identity(variant))[1]);
  }
});

test("no two installs share a port, a directory, an app id, a package, an executable or a tag", () => {
  const installs = [NORMAL, ...VARIANTS.map((slug) => identity(slug))];
  for (const field of ["publicPort", "dirName", "appId", "packageName", "productFilename", "productName", "artifactBase", "tagPrefix"]) {
    // Lower-cased: Windows and macOS file systems do not tell AutoGPT from autogpt.
    const values = installs.map((install) => String(install[field]).toLowerCase());
    assert.equal(new Set(values).size, installs.length, `two installs share a ${field}: ${values.join(", ")}`);
  }
});

test("the experimental public port stays fixed and distinct from the standard app", () => {
  assert.equal(identity("experimental").publicPort, 25500);
  assert.notEqual(identity("experimental").publicPort, NORMAL.publicPort);
});

test("each install has a data directory of its own on every system", () => {
  for (const [platform, env] of [
    ["win32", { LOCALAPPDATA: "C:\\Users\\u\\AppData\\Local" }],
    ["darwin", {}],
    ["linux", { XDG_DATA_HOME: "/x/share" }],
  ]) {
    const dirs = [NORMAL, ...VARIANTS.map((slug) => identity(slug))].map((install) =>
      defaultDataDir(platform, env, install).toLowerCase(),
    );
    assert.equal(new Set(dirs).size, dirs.length, platform);
  }
  const experimental = defaultDataDir("linux", { XDG_DATA_HOME: "/x/share" }, identity("experimental"));
  assert.equal(experimental, path.join("/x/share", "AutoGPT-experimental"));
});

test("a data directory named in the environment is the normal app's; a variant takes the one next to it", () => {
  // One variable for the whole user or shell: every install sees the same value.
  const moved = path.join(os.tmpdir(), "elsewhere", "data");
  const env = { AUTOGPT_DESKTOP_DATA_DIR: moved };
  assert.equal(defaultDataDir(process.platform, env), moved);
  assert.equal(defaultDataDir(process.platform, env, NORMAL), moved);
  assert.equal(defaultDataDir(process.platform, env, identity("experimental")), `${moved}-experimental`);
  assert.equal(defaultDataDir(process.platform, { AUTOGPT_DESKTOP_DATA_DIR: `${moved}${path.sep}` }, identity("experimental")), `${moved}-experimental`);
  const dirs = [NORMAL, ...VARIANTS.map((slug) => identity(slug))].map((install) =>
    defaultDataDir(process.platform, env, install).toLowerCase(),
  );
  assert.equal(new Set(dirs).size, dirs.length);
  // The top of a disk has no folder next to it.
  const root = path.parse(os.tmpdir()).root;
  assert.equal(defaultDataDir(process.platform, { AUTOGPT_DESKTOP_DATA_DIR: root }, identity("experimental")), path.join(root, "AutoGPT-experimental"));
  // The normal app's value is passed through as it was given, as before.
  assert.equal(defaultDataDir("linux", { AUTOGPT_DESKTOP_DATA_DIR: "relative/dir/" }), "relative/dir/");
});

function scratchDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "autogpt-data-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

test("a data directory belongs to the install that used it first", (t) => {
  const experimental = identity("experimental");
  for (const [first, second] of [
    [NORMAL, experimental],
    [experimental, NORMAL],
  ]) {
    const dir = path.join(scratchDir(t), "data");
    assert.equal(claimDataDir(dir, first), first.dirName);
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(dir, INSTALL_STAMP), "utf8")), { install: first.dirName });
    // Its own again on every later start, and never the other's.
    assert.equal(claimDataDir(dir, first), first.dirName);
    assert.equal(claimDataDir(dir, second), first.dirName);
    assert.equal(claimDataDir(dir, first), first.dirName);
    assert.deepEqual(fs.readdirSync(dir), [INSTALL_STAMP]);
  }
});

test("a data directory from before installs left their name in it is the normal app's", (t) => {
  for (const used of ["config", "postgres"]) {
    const dir = scratchDir(t);
    fs.mkdirSync(path.join(dir, used));
    assert.equal(claimDataDir(dir, identity("experimental")), NORMAL.dirName);
    assert.ok(!fs.existsSync(path.join(dir, INSTALL_STAMP)), "a variant wrote into the normal app's data");
    assert.equal(claimDataDir(dir, NORMAL), NORMAL.dirName);
    assert.equal(claimDataDir(dir, identity("experimental")), NORMAL.dirName);
  }
  // Electron's profile can be in the directory before the app has run (macOS).
  const fresh = scratchDir(t);
  fs.mkdirSync(path.join(fresh, "Cache"));
  assert.equal(claimDataDir(fresh, identity("experimental")), "AutoGPT-experimental");
});

test("a name file that names nobody is replaced, and never by a variant on the normal app's data", (t) => {
  const dir = scratchDir(t);
  fs.writeFileSync(path.join(dir, INSTALL_STAMP), "");
  assert.equal(claimDataDir(dir, identity("experimental")), "AutoGPT-experimental");
  assert.equal(claimDataDir(dir, NORMAL), "AutoGPT-experimental");
  const used = scratchDir(t);
  fs.mkdirSync(path.join(used, "postgres"));
  fs.writeFileSync(path.join(used, INSTALL_STAMP), "{");
  assert.equal(claimDataDir(used, identity("experimental")), NORMAL.dirName);
  assert.equal(fs.readFileSync(path.join(used, INSTALL_STAMP), "utf8"), "{");
  assert.equal(claimDataDir(used, NORMAL), NORMAL.dirName);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(used, INSTALL_STAMP), "utf8")), { install: "AutoGPT" });
});

test("the shell starts nothing on a data directory that is another install's", () => {
  const main = fs.readFileSync(path.join(DESKTOP, "src", "main.js"), "utf8");
  const boot = main.slice(main.indexOf("function boot() {"), main.indexOf("function startRuntime() {"));
  const asked = boot.indexOf("foreignData = dataOfAnotherInstall();");
  assert.ok(asked > 0, "main.js no longer asks whose the data directory is");
  // Before anything is written to the directory, and instead of the runtime.
  assert.ok(asked < boot.indexOf("fs.mkdirSync(logsDir"));
  assert.match(boot, /if \(!foreignData\) \{\s+fs\.mkdirSync\(logsDir/);
  assert.match(boot, /if \(foreignData\) report\(\{ event: "error", fatal: true, message: foreignData \}\);\s+else startRuntime\(\);/);
  assert.ok(boot.includes("claimDataDir(dataDir, install, {"));
  assert.ok(boot.includes("if (owner === install.dirName) return null;"));
  // The one other thing the shell writes there itself.
  assert.match(main, /function resetOwnerPassword\(\) \{\s+if \(foreignData\) return focusWindow\(\);/);
});

test("the runtime is told the port to prefer and the name to keep its links under", () => {
  assert.deepEqual(runtimeEnvironment(NORMAL), {
    AUTOGPT_DESKTOP_PUBLIC_PORT: "18473",
    AUTOGPT_DESKTOP_INSTALL_NAME: "AutoGPT",
  });
  assert.deepEqual(runtimeEnvironment(identity("experimental")), {
    AUTOGPT_DESKTOP_PUBLIC_PORT: "25500",
    AUTOGPT_DESKTOP_INSTALL_NAME: "AutoGPT-experimental",
  });
});

function fakeApp({ switches = [] } = {}) {
  const calls = [];
  return {
    calls,
    setName: (name) => calls.push(["setName", name]),
    setPath: (name, value) => calls.push(["setPath", name, value]),
    getPath: (name) => `/${name.toLowerCase()}`,
    commandLine: { hasSwitch: (name) => switches.includes(name) },
  };
}

test("a variant has its own profile, which is also where Electron keeps the single-instance lock", () => {
  const app = fakeApp();
  claim(app, identity("experimental"));
  assert.deepEqual(app.calls, [
    ["setName", "AutoGPT (experimental)"],
    ["setPath", "userData", path.join("/appdata", "AutoGPT-experimental")],
  ]);
});

test("a profile given on the command line is left alone", () => {
  const app = fakeApp({ switches: ["user-data-dir"] });
  claim(app, identity("experimental"));
  assert.deepEqual(app.calls, [["setName", "AutoGPT (experimental)"]]);
});

test("the shell claims its identity before it asks for the single-instance lock", () => {
  const main = fs.readFileSync(path.join(DESKTOP, "src", "main.js"), "utf8");
  const claimed = main.indexOf("claim(app, install);");
  assert.ok(claimed > 0, "main.js no longer calls claim(app, install)");
  assert.ok(claimed < main.indexOf("app.requestSingleInstanceLock()"));
  assert.ok(main.indexOf("const install = identity(") < main.indexOf("defaultDataDir("), "the data directory is the install's own");
  assert.ok(main.includes("defaultDataDir(process.platform, process.env, install)"));
  assert.ok(main.includes("...runtimeEnvironment(install),"));
});

test("a variant's windows say which app they are", () => {
  const experimental = identity("experimental");
  assert.equal(windowTitle(experimental, "AutoGPT"), "AutoGPT (experimental)");
  assert.equal(windowTitle(experimental, ""), "AutoGPT (experimental)");
  assert.equal(windowTitle(experimental, "AutoGPT Platform"), "AutoGPT Platform - AutoGPT (experimental)");
  assert.equal(windowTitle(experimental, "Library - AutoGPT (experimental)"), "Library - AutoGPT (experimental)");
});

test("an installed app is the variant it was built as, whatever the environment says", () => {
  const env = { [VARIANT_VARIABLE]: "experimental" };
  const built = { autogptDesktop: { macDeveloperId: false, variant: "experimental" } };
  const normal = { autogptDesktop: { macDeveloperId: false } };
  assert.equal(variantOf({ isPackaged: true, manifest: normal, env }), "");
  assert.equal(variantOf({ isPackaged: true, manifest: {}, env }), "");
  assert.equal(variantOf({ isPackaged: true, manifest: built, env: { [VARIANT_VARIABLE]: "ignored" } }), "experimental");
  assert.equal(variantOf({ isPackaged: true, manifest: built, env: {} }), "experimental");
  // From the source tree (`npm start`) the variable chooses.
  assert.equal(variantOf({ isPackaged: false, manifest: {}, env }), "experimental");
  assert.equal(variantOf({ isPackaged: false, manifest: {}, env: {} }), "");
  assert.equal(variantOf({ isPackaged: false, manifest: {} }), "");
});

test("a build with the variable set is that variant's, in every name electron-builder uses", () => {
  const config = configured({ AUTOGPT_DESKTOP_VARIANT: "experimental", AUTOGPT_DESKTOP_VERSION: "1.4.0" });
  assert.equal(config.appId, "co.agpt.autogpt.desktop.experimental");
  assert.equal(config.productName, "AutoGPT (experimental)");
  // The executable, the macOS bundle and the Linux binary.
  assert.equal(config.executableName, "autogpt-experimental");
  // The package: the Windows install directory, the .deb's name, the
  // updater's cache directory. And what the installed app reads back.
  assert.deepEqual(config.extraMetadata, {
    version: "1.4.0",
    name: "autogpt-experimental",
    productName: "AutoGPT (experimental)",
    autogptDesktop: { macDeveloperId: false, variant: "experimental" },
  });
  assert.equal(variantOf({ isPackaged: true, manifest: config.extraMetadata, env: {} }), "experimental");
  assert.equal(config.nsis.artifactName, "AutoGPT-experimental-Setup-${version}-${arch}.${ext}");
  assert.equal(config.mac.artifactName, "AutoGPT-experimental-${version}-${arch}.${ext}");
  assert.equal(config.appImage.artifactName, "AutoGPT-experimental-${arch}.${ext}");
  assert.equal(config.deb.artifactName, "AutoGPT-experimental-${version}-${arch}.${ext}");
  // Everything else is the normal app's configuration.
  const normal = configured({ AUTOGPT_DESKTOP_VERSION: "1.4.0" });
  const same = (key) => assert.deepEqual(config[key], normal[key], key);
  for (const key of ["directories", "files", "extraResources", "asar", "npmRebuild", "win", "linux", "detectUpdateChannel"]) {
    same(key);
  }
});

test("electron-builder accepts the names a variant is built with", () => {
  // Its AppImage launcher refuses an executable or product file name with
  // anything but letters, digits, dots, hyphens, underscores and spaces.
  for (const slug of VARIANTS) {
    assert.match(identity(slug).productFilename, /^[\p{L}\p{N}._\- ]+$/u);
    assert.match(identity(slug).packageName, /^[a-z0-9][a-z0-9+.-]+$/, "a Debian package name");
  }
});

test("a build with an unsupported distribution is refused", () => {
  for (const slug of ["voice", "experimental-2", "Experimental", "experimental 2", "../x", "x".repeat(25)]) {
    assert.throws(() => configured({ AUTOGPT_DESKTOP_VARIANT: slug }), /AUTOGPT_DESKTOP_VARIANT must be/);
  }
});

// --- updates never cross ------------------------------------------------------

test("a variant is built with no way to reach the repository's latest release", () => {
  const config = configured({ AUTOGPT_DESKTOP_VARIANT: "experimental", AUTOGPT_DESKTOP_VERSION: "1.4.0" });
  // Its own release, and nothing that follows "latest".
  assert.deepEqual(config.publish, [
    {
      provider: "generic",
      url: "https://github.com/Significant-Gravitas/AutoGPT/releases/download/desktop-experimental-v1.4.0",
      useMultipleRangeRequest: false,
    },
  ]);
  assert.deepEqual(config.publish, [releaseFeed("desktop-experimental-v1.4.0")]);
  assert.equal(config.publish[0].url, feedUrl(releaseTag(identity("experimental"), "1.4.0")));
  const development = configured({ AUTOGPT_DESKTOP_VARIANT: "experimental" });
  assert.equal(development.publish[0].provider, "generic");
  assert.ok(development.publish[0].url.endsWith("/desktop-experimental-v0.0.0-dev.0"));
});

test("the release page of a variant's version is its own tag", () => {
  const experimental = identity("experimental");
  assert.equal(releaseUrl("1.4.0", experimental), "https://github.com/Significant-Gravitas/AutoGPT/releases/tag/desktop-experimental-v1.4.0");
  assert.equal(releaseUrl("1.4.0"), "https://github.com/Significant-Gravitas/AutoGPT/releases/tag/desktop-v1.4.0");
  assert.equal(releaseUrl(null), "https://github.com/Significant-Gravitas/AutoGPT/releases/latest");
  // "latest" is the normal app's.
  assert.equal(releaseUrl(null, experimental), "https://github.com/Significant-Gravitas/AutoGPT/releases");
});

test("a variant picks the highest version among the tags that are exactly its own", () => {
  const experimental = identity("experimental");
  const releases = [
    { tag_name: "desktop-v9.0.0", prerelease: false },
    { tag_name: "desktop-experimental-2-v8.0.0", prerelease: true },
    { tag_name: "desktop-experimental-v1.10.0", prerelease: true },
    { tag_name: "desktop-experimental-v1.9.0", prerelease: true },
    { tag_name: "desktop-experimental-v1.2.3", prerelease: true },
    { tag_name: "desktop-lab-v7.0.0", prerelease: true },
    { tag_name: "v0.4.7", prerelease: false },
  ];
  assert.deepEqual(newestRelease(releases, experimental), { tag: "desktop-experimental-v1.10.0", version: "1.10.0" });
  // Not the first one listed: GitHub lists by date, and an older line can
  // be released after a newer one.
  assert.deepEqual(newestRelease([...releases].reverse(), experimental), { tag: "desktop-experimental-v1.10.0", version: "1.10.0" });
  assert.equal(newestRelease([], experimental), null);
});

test("a variant is never offered the normal app's release or another variant's", () => {
  const others = [
    "desktop-v9.0.0",
    "desktop-experimental-2-v9.0.0",
    "desktop-experimental-v2-v9.0.0",
    "desktop-experimental-v-v9.0.0",
    "desktop-experimental-v9.0.0-v9.0.0",
    "desktop-experimentalv9.0.0",
    "desktop-experimental-9.0.0",
    "desktop-Voice-v9.0.0",
    "xdesktop-experimental-v9.0.0",
    "experimental-v9.0.0",
    "v9.0.0",
    "9.0.0",
  ].map((tag_name) => ({ tag_name, prerelease: true }));
  assert.equal(newestRelease(others, identity("experimental")), null);
  // The experimental distribution never takes the standard app's release.
  for (const slug of VARIANTS) {
    const foreign = [NORMAL, ...VARIANTS.filter((other) => other !== slug).map((other) => identity(other))].map(
      (install) => ({ tag_name: releaseTag(install, "9.0.0"), prerelease: true }),
    );
    assert.equal(newestRelease(foreign, identity(slug)), null, slug);
    const own = { tag_name: releaseTag(identity(slug), "1.0.0"), prerelease: true };
    assert.equal(newestRelease([...foreign, own], identity(slug)).tag, own.tag_name, slug);
  }
});

test("a draft, a version with a pre-release part and a tag that is no version are offered to nobody", () => {
  const experimental = identity("experimental");
  const releases = [
    { tag_name: "desktop-experimental-v3.0.0", draft: true },
    { tag_name: "desktop-experimental-v2.0.0-rc.1" },
    { tag_name: "desktop-experimental-v2.0" },
    { tag_name: "desktop-experimental-v02.0.0" },
    { tag_name: "desktop-experimental-v2.0.0+build" },
    { tag_name: "desktop-experimental-v2.0.0 " },
    { tag_name: null },
    null,
    {},
    { tag_name: "desktop-experimental-v1.0.0" },
  ];
  assert.deepEqual(newestRelease(releases, experimental), { tag: "desktop-experimental-v1.0.0", version: "1.0.0" });
});

test("an update is this app's only if every file in it is this app's installer", () => {
  const installs = [NORMAL, ...VARIANTS.map((slug) => identity(slug))];
  for (const version of ["1.4.0", "1.4.0-rc.1", "10.20.30"]) {
    for (const owner of installs) {
      for (const name of releaseFiles(owner, version)) {
        const accepting = installs.filter((install) => isOwnUpdateFile(install, version, name));
        assert.deepEqual(accepting, [owner], `${name} is accepted by ${accepting.map((a) => a.productName)}`);
      }
    }
  }
  assert.ok(isOwnUpdateFile(NORMAL, "1.4.0", "AutoGPT-Setup-1.4.0-x64.exe"));
  // Another version's file, an address instead of a name, a name that only
  // starts right.
  for (const name of [
    "AutoGPT-Setup-1.4.1-x64.exe",
    "AutoGPT-Setup-1x4x0-x64.exe",
    "https://example.com/AutoGPT-Setup-1.4.0-x64.exe",
    "../AutoGPT-Setup-1.4.0-x64.exe",
    "x/AutoGPT-Setup-1.4.0-x64.exe",
    "AutoGPT-Setup-1.4.0-x64.exe.bat",
    "autogpt-Setup-1.4.0-x64.exe",
    "",
    null,
    undefined,
  ]) {
    assert.ok(!isOwnUpdateFile(NORMAL, "1.4.0", name), String(name));
  }
  assert.ok(!isOwnUpdateFile(NORMAL, "", "AutoGPT-x86_64.AppImage"));
});

// electron-updater's autoUpdater, as far as updater.js uses it for a variant.
class FakeAutoUpdater extends EventEmitter {
  constructor() {
    super();
    this.log = [];
  }

  setFeedURL(options) {
    this.log.push(["setFeedURL", options]);
  }

  checkForUpdates() {
    this.log.push(["checkForUpdates"]);
    return Promise.resolve(null);
  }

  downloadUpdate() {
    this.log.push(["downloadUpdate"]);
    return Promise.resolve([]);
  }
}

function variantUpdater(findRelease, id = identity("experimental")) {
  const autoUpdater = new FakeAutoUpdater();
  const warnings = [];
  const logger = { info() {}, warn: (text) => warnings.push(text), error: (text) => warnings.push(text) };
  const updater = createUpdater({ autoUpdater, mode: "install", identity: id, findRelease, logger });
  return { autoUpdater, updater, warnings };
}

test("a variant reads one release, its own newest, and never asks without having named it", async () => {
  const { autoUpdater, updater } = variantUpdater(async () => ({ tag: "desktop-experimental-v1.5.0", version: "1.5.0" }));
  await updater.check();
  assert.deepEqual(autoUpdater.log, [
    [
      "setFeedURL",
      {
        provider: "generic",
        url: "https://github.com/Significant-Gravitas/AutoGPT/releases/download/desktop-experimental-v1.5.0",
        useMultipleRangeRequest: false,
      },
    ],
    ["checkForUpdates"],
  ]);
});

test("a variant with no release of its own asks nothing and is up to date", async () => {
  const { autoUpdater, updater } = variantUpdater(async () => null);
  await updater.check();
  assert.deepEqual(autoUpdater.log, []);
  assert.deepEqual(updater.state(), { phase: "idle", version: null, checked: true });
});

test("a variant that cannot list the releases asks nothing, and tries again next time", async () => {
  let calls = 0;
  const { autoUpdater, updater, warnings } = variantUpdater(async () => {
    calls += 1;
    throw new Error("https://api.github.com/... answered 403");
  });
  await updater.check();
  assert.deepEqual(autoUpdater.log, []);
  assert.deepEqual(updater.state(), { phase: "idle", version: null, checked: false });
  assert.match(warnings.join("\n"), /answered 403/);
  await updater.check();
  assert.equal(calls, 2);
});

test("a variant cannot be given an updater that would follow the normal app's releases", () => {
  const autoUpdater = new FakeAutoUpdater();
  assert.throws(
    () => createUpdater({ autoUpdater, mode: "install", identity: identity("experimental") }),
    /a variant needs findRelease/,
  );
});

test("the normal app never lists releases: it asks electron-updater, as before", async () => {
  const autoUpdater = new FakeAutoUpdater();
  const findRelease = () => assert.fail("the normal app must not look for a variant's releases");
  const updater = createUpdater({ autoUpdater, mode: "install", findRelease });
  await updater.check();
  assert.deepEqual(autoUpdater.log, [["checkForUpdates"]]);
});

test("the normal app refuses a release whose files are a variant's, before downloading anything", () => {
  const autoUpdater = new FakeAutoUpdater();
  const errors = [];
  const logger = { info() {}, warn() {}, error: (text) => errors.push(text) };
  const updater = createUpdater({ autoUpdater, mode: "install", logger });
  const files = releaseFiles(identity("experimental"), "9.0.0").map((url) => ({ url, sha512: "x" }));
  autoUpdater.emit("update-available", { version: "9.0.0", files, path: files[0].url });
  assert.equal(updater.state().phase, "idle");
  assert.deepEqual(autoUpdater.log, []);
  assert.match(errors.join("\n"), /Refused version 9\.0\.0: its files are not AutoGPT's/);
  // Its own release is taken as before.
  const own = releaseFiles(NORMAL, "9.0.0").map((url) => ({ url, sha512: "x" }));
  autoUpdater.emit("update-available", { version: "9.0.0", files: own, path: own[0].url });
  assert.equal(updater.state().phase, "downloading");
  assert.deepEqual(autoUpdater.log, [["downloadUpdate"]]);
});

test("a variant refuses the normal app's release and another variant's", () => {
  for (const other of [NORMAL, { artifactBase: "AutoGPT-foreign", productName: "foreign app" }]) {
    const { autoUpdater, updater } = variantUpdater(async () => null);
    const files = releaseFiles(other, "9.0.0").map((url) => ({ url, sha512: "x" }));
    autoUpdater.emit("update-available", { version: "9.0.0", files });
    assert.equal(updater.state().phase, "idle", other.productName);
    // One foreign file among its own is enough to refuse.
    const mixed = [...releaseFiles(identity("experimental"), "9.0.0"), files[0].url].map((url) => ({ url, sha512: "x" }));
    autoUpdater.emit("update-available", { version: "9.0.0", files: mixed });
    assert.equal(updater.state().phase, "idle", other.productName);
    autoUpdater.emit("update-available", { version: "9.0.0", path: files[0].url });
    assert.equal(updater.state().phase, "idle", other.productName);
    assert.deepEqual(autoUpdater.log, []);
  }
});

test("where a variant only announces, a foreign release is not announced either", () => {
  const autoUpdater = new FakeAutoUpdater();
  const updater = createUpdater({ autoUpdater, mode: "notify", identity: identity("experimental"), findRelease: async () => null });
  autoUpdater.emit("update-available", { version: "9.0.0", files: [{ url: "AutoGPT-9.0.0-amd64.deb" }] });
  assert.equal(updater.state().phase, "idle");
  autoUpdater.emit("update-available", { version: "9.0.0", files: [{ url: "AutoGPT-experimental-9.0.0-amd64.deb" }] });
  assert.deepEqual(updater.state(), { phase: "available", version: "9.0.0", checked: true });
});

test("the list of releases is read page by page, and not for ever", async () => {
  const experimental = identity("experimental");
  const page = (count, tag) => Array.from({ length: count }, () => ({ tag_name: tag }));
  const answers = {
    1: page(100, "desktop-v1.0.0"),
    2: [...page(99, "desktop-v1.0.0"), { tag_name: "desktop-experimental-v1.2.0" }],
    3: page(7, "desktop-experimental-v1.1.0"),
    4: page(100, "desktop-experimental-v9.9.9"),
  };
  const asked = [];
  const fetchJson = async (url) => {
    asked.push(url);
    return answers[new URL(url).searchParams.get("page")];
  };
  assert.deepEqual(await variantReleaseFinder({ id: experimental, fetchJson })(), {
    tag: "desktop-experimental-v1.2.0",
    version: "1.2.0",
  });
  assert.deepEqual(asked, [1, 2, 3].map((n) => `https://api.github.com/repos/Significant-Gravitas/AutoGPT/releases?per_page=100&page=${n}`));

  const endless = [];
  await variantReleaseFinder({
    id: experimental,
    fetchJson: async (url) => {
      endless.push(url);
      return page(100, "desktop-v1.0.0");
    },
  })();
  assert.equal(endless.length, 5);

  await assert.rejects(
    variantReleaseFinder({ id: experimental, fetchJson: async () => ({ message: "API rate limit exceeded" }) })(),
    /did not answer with a list/,
  );
});

// --- what electron-updater itself asks for ------------------------------------
//
// The two providers are run as they are installed, against a recorded
// network: this is where "the normal app cannot pick a variant's release" and
// "a variant cannot pick anybody else's" stop being a reading of their code.

const ATOM = (tags) => `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
${tags
  .map(
    (tag) => `<entry>
  <link rel="alternate" type="text/html" href="https://github.com/Significant-Gravitas/AutoGPT/releases/tag/${tag}"/>
  <title>${tag}</title>
  <content type="html">notes</content>
</entry>`,
  )
  .join("\n")}
</feed>`;

function metadataFor(id, version) {
  const [installer] = releaseFiles(id, version);
  return `version: ${version}\nfiles:\n  - url: ${installer}\n    sha512: abc\n    size: 1\npath: ${installer}\nsha512: abc\n`;
}

// What electron-updater's executor is: asked for a URL, answers with a body.
function recordedNetwork(answer) {
  const asked = [];
  return {
    asked,
    executor: {
      request(options) {
        const url = `https://${options.hostname}${options.path}`;
        asked.push(url);
        return Promise.resolve(answer(url));
      },
    },
  };
}

function provider(t, publish, network) {
  if (!fs.existsSync(UPDATER)) return t.skip("electron-updater is not installed (npm ci)");
  const { createClient } = require(path.join(UPDATER, "providerFactory.js"));
  // What src/updater.js configures: no channel, no pre-releases.
  const updater = { allowPrerelease: false, channel: null, fullChangelog: false, isAddNoCacheQuery: false };
  return createClient(publish, updater, { executor: network.executor, platform: "win32", isUseMultipleRangeRequest: true });
}

test("electron-updater gives the normal app the release GitHub calls latest, and nothing else", async (t) => {
  const network = recordedNetwork((url) => {
    // Variants were released more recently than the normal app.
    if (url.endsWith("/releases.atom")) return ATOM(["desktop-experimental-v9.9.9", "desktop-lab-v8.0.0", "desktop-v1.4.0"]);
    if (url.endsWith("/releases/latest")) return JSON.stringify({ tag_name: "desktop-v1.4.0" });
    if (url.endsWith("/releases/download/desktop-v1.4.0/latest.yml")) return metadataFor(NORMAL, "1.4.0");
    throw new Error(`unexpected request: ${url}`);
  });
  const client = provider(t, configured().publish[0], network);
  if (!client) return;
  const info = await client.getLatestVersion();
  assert.equal(info.tag, "desktop-v1.4.0");
  assert.equal(info.version, "1.4.0");
  // The tag comes from /releases/latest alone, which GitHub never answers
  // with a pre-release; experimental releases must remain pre-releases.
  assert.deepEqual(network.asked, [
    "https://github.com/Significant-Gravitas/AutoGPT/releases.atom",
    "https://github.com/Significant-Gravitas/AutoGPT/releases/latest",
    "https://github.com/Significant-Gravitas/AutoGPT/releases/download/desktop-v1.4.0/latest.yml",
  ]);
  const [file] = client.resolveFiles(info);
  assert.equal(file.url.href, "https://github.com/Significant-Gravitas/AutoGPT/releases/download/desktop-v1.4.0/AutoGPT-Setup-1.4.0-x64.exe");
});

test("electron-updater gives a variant the one release it was pointed at, and nothing else", async (t) => {
  const experimental = identity("experimental");
  const built = configured({ AUTOGPT_DESKTOP_VARIANT: "experimental", AUTOGPT_DESKTOP_VERSION: "1.4.0" }).publish[0];
  // As built (app-update.yml), and as src/updater.js re-points it.
  for (const [publish, version] of [
    [built, "1.4.0"],
    [releaseFeed("desktop-experimental-v1.5.0"), "1.5.0"],
  ]) {
    const release = `https://github.com/Significant-Gravitas/AutoGPT/releases/download/desktop-experimental-v${version}`;
    const network = recordedNetwork((url) => {
      if (url === `${release}/latest.yml`) return metadataFor(experimental, version);
      throw new Error(`unexpected request: ${url}`);
    });
    const client = provider(t, publish, network);
    if (!client) return;
    const info = await client.getLatestVersion();
    assert.equal(info.version, version);
    assert.deepEqual(network.asked, [`${release}/latest.yml`]);
    const [file] = client.resolveFiles(info);
    assert.equal(file.url.href, `${release}/AutoGPT-experimental-Setup-${version}-x64.exe`);
  }
});

test("a variant downloads only what changed, as the normal app does", (t) => {
  // GitHub's file host answers a request for several ranges with 501.
  // electron-updater's GitHub provider never sends one; its generic provider
  // does unless told not to, and then falls back to the whole installer.
  const network = recordedNetwork(() => "");
  const normal = provider(t, configured().publish[0], network);
  if (!normal) return;
  assert.equal(normal.isUseMultipleRangeRequest, false);
  const built = configured({ AUTOGPT_DESKTOP_VARIANT: "experimental", AUTOGPT_DESKTOP_VERSION: "1.4.0" }).publish[0];
  for (const publish of [built, releaseFeed("desktop-experimental-v1.5.0")]) {
    assert.equal(publish.useMultipleRangeRequest, false);
    assert.equal(provider(t, publish, network).isUseMultipleRangeRequest, false);
  }
  // What it would be without the setting: the test fails if electron-updater
  // stops needing it.
  const { useMultipleRangeRequest: _, ...plain } = built;
  assert.equal(provider(t, plain, network).isUseMultipleRangeRequest, true);
});

test("electron-updater still lets the feed be named while the app runs", (t) => {
  if (!fs.existsSync(UPDATER)) return t.skip("electron-updater is not installed (npm ci)");
  const source = fs.readFileSync(path.join(UPDATER, "AppUpdater.js"), "utf8");
  assert.match(source, /\n {4}setFeedURL\(options\) \{/, "electron-updater no longer has setFeedURL");
  // A feed that was named is used instead of the one in app-update.yml.
  assert.ok(source.includes("this.clientPromise = Promise.resolve(provider);"));
  assert.match(source, /if \(this\.clientPromise == null\) \{\s+this\.clientPromise = this\.configOnDisk/);
});

// --- the installed-app tests ------------------------------------------------------

test("an uninstall entry is an app's own only by its whole name", () => {
  // DisplayName is "<product name> <version>", and "AutoGPT" starts every name.
  const experimental = identity("experimental");
  assert.ok(isOwnUninstallName(NORMAL, "AutoGPT 1.4.0"));
  assert.ok(isOwnUninstallName(NORMAL, "AutoGPT 0.0.0-dev.12"));
  assert.ok(isOwnUninstallName(experimental, "AutoGPT (experimental) 1.4.0"));
  assert.ok(!isOwnUninstallName(NORMAL, "AutoGPT (experimental) 1.4.0"));
  assert.ok(!isOwnUninstallName(experimental, "AutoGPT 1.4.0"));
  assert.ok(!isOwnUninstallName(experimental, "AutoGPT (experimental-2) 1.4.0"));
  assert.ok(!isOwnUninstallName(NORMAL, "AutoGPT"));
  assert.ok(!isOwnUninstallName(NORMAL, undefined));
});
