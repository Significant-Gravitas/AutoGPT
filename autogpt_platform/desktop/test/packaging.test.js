"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { test } = require("node:test");

const { signMacApp } = require("../build/mac_sign");

const DESKTOP = path.join(__dirname, "..");
const CONFIG = path.join(DESKTOP, "electron-builder.config.js");
const SETTINGS = [
  "AUTOGPT_DESKTOP_VERSION",
  "AUTOGPT_DESKTOP_OUTPUT",
  "AUTOGPT_DESKTOP_VARIANT",
  "AUTOGPT_DESKTOP_MAC_SIGN",
  "AUTOGPT_DESKTOP_WIN_SIGN",
  "AZURE_SIGN_ENDPOINT",
  "AZURE_SIGN_ACCOUNT",
  "AZURE_SIGN_PROFILE",
  "AZURE_SIGN_PUBLISHER",
];

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

function onPlatform(t, platform) {
  const original = Object.getOwnPropertyDescriptor(process, "platform");
  Object.defineProperty(process, "platform", { value: platform });
  t.after(() => Object.defineProperty(process, "platform", original));
}

test("with nothing set, macOS is ad-hoc signed and Windows unsigned, as before there was a certificate", () => {
  const config = configured();
  assert.deepEqual(config.mac, {
    category: "public.app-category.productivity",
    minimumSystemVersion: "15.0",
    artifactName: "AutoGPT-${version}-${arch}.${ext}",
    gatekeeperAssess: false,
    target: [{ target: "dmg", arch: ["arm64"] }],
    identity: "-",
    hardenedRuntime: false,
    signIgnore: ["/Contents/Resources/runtime/site\\/claude_agent_sdk\\/_bundled\\/claude$"],
  });
  // electron-builder tests each of these against a file's full path.
  const [ignored] = config.mac.signIgnore.map((pattern) => new RegExp(pattern));
  const runtime = "/Users/me/dist/mac-arm64/AutoGPT.app/Contents/Resources/runtime";
  assert.ok(ignored.test(`${runtime}/site/claude_agent_sdk/_bundled/claude`));
  assert.ok(!ignored.test(`${runtime}/site/claude_agent_sdk/_bundled/claude.md`));
  assert.ok(!ignored.test(`${runtime}/python/bin/python3.13`));
  assert.deepEqual(config.win, { target: [{ target: "nsis", arch: ["x64"] }] });
  assert.equal(config.forceCodeSigning, false);
  assert.equal(config.directories.output, "dist");
  assert.deepEqual(config.extraMetadata, { autogptDesktop: { macDeveloperId: false } });
});

test("package.json no longer carries a second configuration", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(DESKTOP, "package.json"), "utf8"));
  assert.equal(manifest.build, undefined);
  for (const script of [manifest.scripts.dist, manifest.scripts["dist:dir"]]) {
    assert.match(script, /--config electron-builder\.config\.js/);
    assert.match(script, /--publish never/);
  }
});

test("the version comes from the environment and must be a version", () => {
  assert.equal(configured({ AUTOGPT_DESKTOP_VERSION: "1.4.0" }).extraMetadata.version, "1.4.0");
  assert.equal(configured({ AUTOGPT_DESKTOP_VERSION: "1.4.0-rc.1" }).extraMetadata.version, "1.4.0-rc.1");
  assert.equal(configured({ AUTOGPT_DESKTOP_VERSION: "0.0.0-dev.412" }).extraMetadata.version, "0.0.0-dev.412");
  for (const good of ["0.0.1-dev.0", "10.20.30", "1.4.0-rc.10", "1.4.0-0a", "1.4.0-alpha-1.x"]) {
    assert.equal(configured({ AUTOGPT_DESKTOP_VERSION: good }).extraMetadata.version, good);
  }
  // Leading zeros are not semantic versioning: electron-updater refuses to
  // start in an app with such a version, and to read one from latest.yml.
  const leadingZeros = ["01.4.0", "1.04.0", "1.4.00", "1.4.0-rc.01", "1.4.0-01"];
  for (const bad of ["v1.4.0", "1.4", "desktop-v1.4.0", "1.4.0 ", "1.4.0+build", "1.4.0-", "1.4.0-rc..1", ...leadingZeros]) {
    assert.throws(() => configured({ AUTOGPT_DESKTOP_VERSION: bad }), /AUTOGPT_DESKTOP_VERSION/, bad);
  }
});

test("a Developer ID build is hardened, notarized, and signed by build/mac_sign.js", (t) => {
  onPlatform(t, "darwin");
  const config = configured({ AUTOGPT_DESKTOP_MAC_SIGN: "developer-id" });
  assert.equal(config.mac.identity, undefined);
  assert.equal(config.mac.hardenedRuntime, true);
  assert.equal(config.mac.notarize, true);
  assert.equal(config.mac.sign, signMacApp);
  assert.ok(fs.existsSync(config.mac.entitlements));
  assert.ok(fs.existsSync(config.mac.entitlementsInherit));
  // Squirrel.Mac updates from the zip.
  assert.deepEqual(config.mac.target.map((target) => target.target), ["dmg", "zip"]);
  assert.equal(config.extraMetadata.autogptDesktop.macDeveloperId, true);
  assert.equal(config.forceCodeSigning, true);
});

test("only a Developer ID build on a Mac looks at the packed app", (t) => {
  assert.equal(configured().artifactBuildStarted, undefined);
  onPlatform(t, "linux");
  assert.equal(configured({ AUTOGPT_DESKTOP_MAC_SIGN: "developer-id" }).artifactBuildStarted, undefined);
});

test("a Developer ID build looks at the packed app before making an installer from it", async (t) => {
  // electron-builder ignores forceCodeSigning on macOS when `sign` is a function.
  onPlatform(t, "darwin");
  const hook = configured({ AUTOGPT_DESKTOP_MAC_SIGN: "developer-id" }).artifactBuildStarted;
  assert.equal(typeof hook, "function");
  // No codesign here (or no app there): either way the build is refused, and
  // the disk image and the zip get the same answer from one look. An output
  // directory of its own: dist/ may hold an app someone built.
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "no-app-"));
  t.after(() => fs.rmSync(output, { recursive: true, force: true }));
  const dmg = hook({ file: path.join(output, "AutoGPT-1.4.0-arm64.dmg") });
  const zip = hook({ file: path.join(output, "AutoGPT-1.4.0-arm64.zip") });
  assert.equal(dmg, zip);
  await assert.rejects(dmg, /A Developer ID build was asked for, but AutoGPT\.app is not signed/);
});

test("the configuration loads where nothing has been installed", () => {
  // The unit tests run before `npm install` on the build machines.
  const { execFileSync } = require("node:child_process");
  const bare = `
    const Module = require("node:module");
    const resolve = Module._resolveFilename;
    Module._resolveFilename = function (request, ...rest) {
      if (!request.startsWith(".") && !request.startsWith("node:") && !require("node:path").isAbsolute(request) && !Module.builtinModules.includes(request)) {
        throw Object.assign(new Error("Cannot find module '" + request + "'"), { code: "MODULE_NOT_FOUND" });
      }
      return resolve.call(this, request, ...rest);
    };
    const config = require(process.argv[1]);
    require("node:fs").writeSync(1, config.appId);
  `;
  const appId = execFileSync(process.execPath, ["-e", bare, path.join(DESKTOP, "electron-builder.config.js")], {
    cwd: DESKTOP,
    encoding: "utf8",
    env: { ...process.env, AUTOGPT_DESKTOP_VARIANT: "", AUTOGPT_DESKTOP_MAC_SIGN: "", AUTOGPT_DESKTOP_WIN_SIGN: "" },
  });
  assert.equal(appId, "co.agpt.autogpt.desktop");
});

// What @electron/osx-sign does to every file of the app, in a process that
// may have fewer files open than the directory holds.
const OPEN_EVERY_FILE = `
  const [, configFile, dir] = process.argv;
  if (configFile) require(configFile);
  const { isBinaryFile } = require("isbinaryfile");
  const fs = require("node:fs"), path = require("node:path");
  const files = fs.readdirSync(dir).map((name) => path.join(dir, name));
  // Written straight to the descriptor: with none to spare, process.stdout
  // cannot even be set up.
  Promise.all(files.map((file) => isBinaryFile(file))).then(
    () => fs.writeSync(1, "opened them all"),
    (error) => { fs.writeSync(1, String(error.code)); process.exit(0); },
  );
`;

function openEveryFile(dir, configFile) {
  const { execFileSync } = require("node:child_process");
  const command = 'ulimit -n 256 && exec "$0" -e "$1" "$2" "$3"';
  return execFileSync("sh", ["-c", command, process.execPath, OPEN_EVERY_FILE, configFile, dir], {
    cwd: DESKTOP,
    encoding: "utf8",
  }).trim();
}

test(
  "signing a bundle of more files than may be open at once waits instead of failing",
  { skip: process.platform === "win32" && "Windows has no such limit, and nothing is signed this way there" },
  (t) => {
    try {
      require.resolve("isbinaryfile");
    } catch {
      return t.skip("npm install has not been run");
    }
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "open-files-"));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    for (let index = 0; index < 1000; index += 1) fs.writeFileSync(path.join(dir, `f${index}`), "text");
    assert.equal(
      openEveryFile(dir, ""),
      "EMFILE",
      "the library that signs no longer fails on its own when it opens more files than are allowed: " +
        "the graceful-fs line at the top of electron-builder.config.js, and this test, can go",
    );
    assert.equal(openEveryFile(dir, path.join(DESKTOP, "electron-builder.config.js")), "opened them all");
  },
);

test("any other value for the macOS setting is the unsigned build", () => {
  for (const value of ["true", "1", "Developer-ID", ""]) {
    const config = configured({ AUTOGPT_DESKTOP_MAC_SIGN: value });
    assert.equal(config.mac.identity, "-");
    assert.equal(config.extraMetadata.autogptDesktop.macDeveloperId, false);
  }
});

test("a signed Windows build leaves the Claude Code CLI as Anthropic shipped it", (t) => {
  onPlatform(t, "win32");
  const pfx = configured({ AUTOGPT_DESKTOP_WIN_SIGN: "pfx" });
  assert.deepEqual(pfx.win.signExts, ["!claude.exe"]);
  assert.equal(pfx.win.azureSignOptions, undefined);
  assert.equal(pfx.forceCodeSigning, true);
});

test("Azure Trusted Signing needs all four of its settings", (t) => {
  onPlatform(t, "win32");
  assert.throws(() => configured({ AUTOGPT_DESKTOP_WIN_SIGN: "azure" }), /AZURE_SIGN_ENDPOINT, AZURE_SIGN_ACCOUNT/);
  const config = configured({
    AUTOGPT_DESKTOP_WIN_SIGN: "azure",
    AZURE_SIGN_ENDPOINT: "https://eus.codesigning.azure.net",
    AZURE_SIGN_ACCOUNT: "account",
    AZURE_SIGN_PROFILE: "profile",
    AZURE_SIGN_PUBLISHER: "CN=Example",
  });
  assert.deepEqual(config.win.azureSignOptions, {
    endpoint: "https://eus.codesigning.azure.net",
    codeSigningAccountName: "account",
    certificateProfileName: "profile",
    publisherName: "CN=Example",
  });
});

test("a signing setting for another system does not force signing here", (t) => {
  onPlatform(t, "linux");
  const config = configured({ AUTOGPT_DESKTOP_MAC_SIGN: "developer-id", AUTOGPT_DESKTOP_WIN_SIGN: "pfx" });
  assert.equal(config.forceCodeSigning, false);
});

test("the AppImage's name has no version in it, so updating keeps the file's name", () => {
  const config = configured({ AUTOGPT_DESKTOP_VERSION: "1.4.0" });
  assert.ok(!config.appImage.artifactName.includes("version"));
  // Every other file of a release is named after its version.
  for (const name of [config.nsis.artifactName, config.mac.artifactName, config.deb.artifactName]) {
    assert.ok(name.includes("${version}"), name);
  }
});

test("update files are always latest*.yml and nothing is published from a build", () => {
  const config = configured({ AUTOGPT_DESKTOP_VERSION: "1.4.0-rc.1" });
  assert.equal(config.detectUpdateChannel, false);
  assert.deepEqual(config.publish, [{ provider: "github", owner: "ntindle", repo: "autogpt" }]);
});

test("the runtime ships inside the app, and the updater ships with the shell", () => {
  const config = configured();
  assert.deepEqual(config.extraResources, [{ from: "build/runtime", to: "runtime", filter: ["**/*"] }]);
  const manifest = JSON.parse(fs.readFileSync(path.join(DESKTOP, "package.json"), "utf8"));
  // devDependencies are not packed into the app.
  assert.ok(manifest.dependencies["electron-updater"], "electron-updater must be a dependency");
  assert.match(manifest.dependencies["electron-updater"], /^\d/, "pin electron-updater to one version");
});

test("nothing is compiled or run from the dependencies while packaging", () => {
  assert.equal(configured().npmRebuild, false);
  // What makes that safe: no package that ships in the app builds itself
  // on install. A native module would need npmRebuild again.
  const lock = JSON.parse(fs.readFileSync(path.join(DESKTOP, "package-lock.json"), "utf8"));
  const building = Object.entries(lock.packages)
    .filter(([, entry]) => entry.hasInstallScript && !entry.dev)
    .map(([name]) => name);
  assert.deepEqual(building, []);
});

test("certificates and passwords never pass through the configuration", () => {
  const source = fs.readFileSync(CONFIG, "utf8");
  const read = [...source.matchAll(/env\.([A-Z_]+)|env\[name\]/g)].map((match) => match[1]).filter(Boolean);
  assert.deepEqual([...new Set(read)].sort(), [
    "AUTOGPT_DESKTOP_MAC_SIGN",
    "AUTOGPT_DESKTOP_OUTPUT",
    "AUTOGPT_DESKTOP_VARIANT",
    "AUTOGPT_DESKTOP_VERSION",
    "AUTOGPT_DESKTOP_WIN_SIGN",
  ]);
});

// --- the Windows installer and other installs ---------------------------------

const INCLUDE = path.join(DESKTOP, "resources", "installer.nsh");
const TEMPLATE = path.join(
  DESKTOP,
  "node_modules",
  "app-builder-lib",
  "templates",
  "nsis",
  "include",
  "allowOnlyOneInstallerInstance.nsh",
);

function macro(source, name) {
  const found = new RegExp(`^!macro ${name}\\b[\\s\\S]*?^!macroend[ \\t]*$`, "m").exec(source);
  return found ? found[0].replace(/[ \t]+$/gm, "") : null;
}

test("every Windows installer closes the programs of its own install only", () => {
  // Install directories are siblings whose names start the same:
  // ...\Programs\autogpt, ...\Programs\autogpt-voice, ...\Programs\autogpt-voice2.
  for (const variant of ["", "voice"]) {
    const config = configured({ AUTOGPT_DESKTOP_VARIANT: variant });
    assert.equal(config.nsis.include, "resources/installer.nsh");
    assert.equal(config.directories.buildResources, "resources");
  }
  const include = fs.readFileSync(INCLUDE, "utf8").replace(/\r\n/g, "\n");
  // This macro is what makes electron-builder's template use the check below.
  const setup = macro(include, "customCheckAppRunning");
  assert.ok(setup.includes('System::Call \'Kernel32::SetEnvironmentVariable(t "AUTOGPT_DESKTOP_INSTALL_DIR", t "$INSTDIR\\")i.r0\''));
  assert.match(setup, /Push \$0\n {2}System::Call/);
  assert.match(setup, /\$\{if\} \$0 == 0\n {4}Pop \$0\n {4}Abort /);
  assert.match(setup, /Pop \$0\n {2}!insertmacro IS_POWERSHELL_AVAILABLE\n {2}!insertmacro OWN_CHECK_APP_RUNNING\n!macroend$/);
  // A program is the install's if its path starts with the directory and a
  // backslash; the directory's name alone is also the start of a sibling's.
  const compared = [...include.matchAll(/^[^#\n]*StartsWith\(([^,]+),/gm)].map((match) => match[1]);
  assert.deepEqual(compared, ["$$env:AUTOGPT_DESKTOP_INSTALL_DIR", "$$env:AUTOGPT_DESKTOP_INSTALL_DIR"]);
  for (const body of ["OWN_FIND_PROCESS", "OWN_KILL_PROCESS", "OWN_CHECK_APP_RUNNING"].map((name) => macro(include, name))) {
    assert.ok(body, "installer.nsh lost one of its macros");
    assert.ok(!/!insertmacro (FIND|KILL)_PROCESS/.test(body), "a check of electron-builder's own is still used");
  }
});

test("Windows installer process queries preserve literal install paths", {
  skip: process.platform !== "win32" && "PowerShell is a Windows installer dependency",
  timeout: 60_000,
}, () => {
  const { spawnSync } = require("node:child_process");
  const include = fs.readFileSync(INCLUDE, "utf8").replace(/\r\n/g, "\n");
  const powershell = path.join(process.env.SystemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  const fakeProcesses = [
    "$ProgressPreference = 'SilentlyContinue'",
    "function Get-CimInstance { param($ClassName) $items = ConvertFrom-Json $env:AUTOGPT_INSTALL_TEST_PROCESSES; foreach ($item in $items) { $item } }",
    "function Stop-Process { param([int]$Id, [switch]$Force) Write-Output $Id }",
  ].join("\n");
  function runMacro(name, directory, processes) {
    const command = macro(include, name).match(/nsExec::Exec `"\$PowerShellPath" -C "([^"]+)"`/)[1]
      .replaceAll("$$", "$")
      .replaceAll("$INSTDIR", directory)
      .replaceAll("$0", "-Force");
    const encoded = Buffer.from(`${fakeProcesses}\n${command}`, "utf16le").toString("base64");
    const result = spawnSync(powershell, ["-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], {
      encoding: "utf8",
      timeout: 5000,
      windowsHide: true,
      env: {
        ...process.env,
        AUTOGPT_DESKTOP_INSTALL_DIR: `${directory}\\`,
        AUTOGPT_INSTALL_TEST_PROCESSES: JSON.stringify(processes),
      },
    });
    assert.ifError(result.error);
    assert.equal(result.stderr.trim(), "", `${name}: ${directory}`);
    return result;
  }
  for (const profile of ["Nick", "John Smith", "O'Brien", "Jürgen"]) {
    const directory = `C:\\Users\\${profile}\\AppData\\Local\\Programs\\autogpt`;
    const own = [
      { Path: `${directory}\\AutoGPT.exe`, ProcessId: 101 },
      { Path: `${directory}\\resources\\runtime\\python\\python.exe`, ProcessId: 102 },
    ];
    const neighbors = [
      { Path: `${directory}-voice\\AutoGPT.exe`, ProcessId: 201 },
      { Path: `${directory}2\\AutoGPT.exe`, ProcessId: 202 },
      { Path: null, ProcessId: 203 },
    ];
    assert.equal(runMacro("OWN_FIND_PROCESS", directory, [...own, ...neighbors]).status, 0);
    assert.equal(runMacro("OWN_FIND_PROCESS", directory, neighbors).status, 1);
    const stopped = runMacro("OWN_KILL_PROCESS", directory, [...own, ...neighbors]);
    assert.equal(stopped.status, 0);
    assert.deepEqual(stopped.stdout.trim().split(/\r?\n/), ["101", "102"]);
  }
});

test("the installer's check is electron-builder's, changed in two places", (t) => {
  if (!fs.existsSync(TEMPLATE)) return t.skip("electron-builder is not installed (npm ci)");
  const template = fs.readFileSync(TEMPLATE, "utf8").replace(/\r\n/g, "\n");
  const include = fs.readFileSync(INCLUDE, "utf8").replace(/\r\n/g, "\n");
  // The template still takes a replacement, and still needs one.
  assert.match(template, /!ifmacrodef customCheckAppRunning\n\s+!insertmacro customCheckAppRunning\n\s+!else/);
  assert.equal(template.split("StartsWith('$INSTDIR', 'CurrentCultureIgnoreCase')").length - 1, 2);
  // What installer.nsh needs from the template and from its include folder.
  assert.ok(macro(template, "IS_POWERSHELL_AVAILABLE"));
  assert.match(template, /!ifmacrondef customCheckAppRunning\n\s+!include "getProcessInfo\.nsh"\n\s+Var pid\n!endif/);
  assert.ok(fs.existsSync(path.join(path.dirname(TEMPLATE), "getProcessInfo.nsh")));
  assert.match(include, /^!include "getProcessInfo\.nsh"\nVar pid$/m);

  const own = (text) =>
    text
      .replaceAll("'$INSTDIR'", () => "$$env:AUTOGPT_DESKTOP_INSTALL_DIR")
      .replace("if ((Get-CimInstance", "if (@(Get-CimInstance")
      .replace(/^!macro _CHECK_APP_RUNNING/, "!macro OWN_CHECK_APP_RUNNING")
      .replace(/!(insert)?macro (FIND|KILL)_PROCESS/g, "!$1macro OWN_$2_PROCESS");
  for (const [theirs, ours] of [
    ["FIND_PROCESS", "OWN_FIND_PROCESS"],
    ["KILL_PROCESS", "OWN_KILL_PROCESS"],
    ["_CHECK_APP_RUNNING", "OWN_CHECK_APP_RUNNING"],
  ]) {
    assert.equal(
      macro(include, ours),
      own(macro(template, theirs)),
      `electron-builder changed ${theirs}: carry the change over to ${ours} in resources/installer.nsh`,
    );
  }
});

// --- how an update removes the old version ------------------------------------

const UNINSTALLER = path.join(DESKTOP, "node_modules", "app-builder-lib", "templates", "nsis", "uninstaller.nsh");
const lines = (text) => text.split("\n").map((line) => line.trim()).filter(Boolean);

test("an update moves the old install away in one rename, to a sibling on the same drive", () => {
  const include = fs.readFileSync(INCLUDE, "utf8").replace(/\r\n/g, "\n");
  const remove = macro(include, "customRemoveFiles");
  assert.ok(remove, "installer.nsh has no customRemoveFiles: every update renames each file of the old version");
  const steps = lines(remove).filter((line) => !line.startsWith("#"));
  const rename = steps.indexOf('Rename "$INSTDIR" "$INSTDIR.old-install"');
  assert.ok(rename > 0, "the whole directory is renamed, beside itself");
  // A directory cannot be renamed while it is somebody's working directory,
  // and the uninstaller's is $INSTDIR.
  assert.ok(steps.slice(0, rename).includes("SetOutPath $TEMP"));
  // What an earlier update left is in the way of the rename.
  assert.ok(steps.slice(0, rename).includes('RMDir /r "$INSTDIR.old-install"'));
  assert.equal(steps[rename - 1], "ClearErrors");
  assert.equal(steps[rename + 1], "${if} ${Errors}");
  // Not into the installer's temporary directory: that may be another drive.
  assert.ok(!/Rename "\$INSTDIR" "\$(PLUGINSDIR|TEMP)/.test(remove));
  // An uninstall that is not an update deletes in place, as it always did.
  assert.deepEqual(steps.slice(-4), ["SetOutPath $TEMP", "RMDir /r $INSTDIR", "${endif}", "!macroend"]);
});

test("an uninstall takes what an update could not finish deleting", () => {
  // A file held by a scanner while the renamed directory was deleted leaves
  // it behind, up to 1.5 GB beside the install, registered nowhere.
  const include = fs.readFileSync(INCLUDE, "utf8").replace(/\r\n/g, "\n");
  const steps = lines(macro(include, "customRemoveFiles")).filter((line) => !line.startsWith("#"));
  const plain = steps.slice(steps.lastIndexOf("${else}"));
  assert.deepEqual(plain.slice(0, 5), [
    "${else}",
    '${if} ${FileExists} "$INSTDIR.old-install\\*.*"',
    'RMDir /r "$INSTDIR.old-install"',
    "ClearErrors",
    "${endif}",
  ]);
  // And the update that could not finish tries once more itself.
  const renamed = steps.indexOf('DetailPrint "Moved the old version away in one piece."');
  assert.ok(renamed > 0);
  assert.deepEqual(steps.slice(renamed + 1, renamed + 4), ['RMDir /r "$INSTDIR.old-install"', "${if} ${Errors}", "Sleep 2000"]);
});

test("the installer that waits for the old version's removal is not standing in its directory", (t) => {
  // Windows refuses to rename a directory that is a program's working
  // directory. electron-updater starts the installer without one, so it has
  // the app's: the install directory, for an app started from its shortcut.
  const include = fs.readFileSync(INCLUDE, "utf8").replace(/\r\n/g, "\n");
  assert.equal(macro(include, "customInit"), "!macro customInit\n  SetOutPath $TEMP\n!macroend");

  const templates = path.join(DESKTOP, "node_modules", "app-builder-lib", "templates", "nsis");
  const updater = path.join(DESKTOP, "node_modules", "electron-updater", "out", "BaseUpdater.js");
  if (!fs.existsSync(templates) || !fs.existsSync(updater)) return t.skip("electron-builder is not installed (npm ci)");
  // The template runs customInit in the installer's .onInit, after the line
  // that would otherwise leave the installer in $INSTDIR, and not in the
  // uninstaller's build.
  const installer = fs.readFileSync(path.join(templates, "installer.nsi"), "utf8").replace(/\r\n/g, "\n");
  const onInit = /^Function \.onInit\n([\s\S]*?)^FunctionEnd$/m.exec(installer);
  assert.ok(onInit, "electron-builder's installer has no .onInit any more: read templates/nsis/installer.nsi");
  const own = onInit[1].indexOf("!ifmacrodef customInit\n      !insertmacro customInit");
  assert.ok(own > onInit[1].indexOf("SetOutPath $INSTDIR"), "customInit no longer runs after .onInit goes to $INSTDIR");
  assert.ok(own > onInit[1].indexOf("!ifdef BUILD_UNINSTALLER") && own > onInit[1].indexOf("!else"));
  // The old version is removed in the install section, which only afterwards
  // goes back to $INSTDIR and does not go there before.
  const section = fs.readFileSync(path.join(templates, "installSection.nsh"), "utf8").replace(/\r\n/g, "\n");
  const removal = section.indexOf("!insertmacro uninstallOldVersion SHELL_CONTEXT");
  assert.ok(removal > 0, "electron-builder removes the old version somewhere else now: read installSection.nsh");
  assert.ok(!section.slice(0, removal).includes("SetOutPath"), "the installer changes directory before the old version is removed");
  assert.ok(section.slice(removal).includes("\nSetOutPath $INSTDIR\n"), "the installer no longer returns to $INSTDIR to write the files");
  // And the updater still starts the installer where the app is.
  const spawn = /async spawnLog\([\s\S]*?\n {4}\}\n/.exec(fs.readFileSync(updater, "utf8"));
  assert.ok(spawn, "electron-updater starts the installer some other way now: read BaseUpdater.js");
  assert.ok(!spawn[0].includes("cwd"), "electron-updater names a working directory now: customInit may no longer be needed");
});

test("when Windows refuses that rename, the removal is electron-builder's own", (t) => {
  if (!fs.existsSync(UNINSTALLER)) return t.skip("electron-builder is not installed (npm ci)");
  const template = fs.readFileSync(UNINSTALLER, "utf8").replace(/\r\n/g, "\n");
  // The template still takes a replacement, and still has the two functions
  // the replacement calls (unused, they would be a warning, which fails the build).
  const own = /!ifmacrodef customRemoveFiles\n\s+!insertmacro customRemoveFiles\n\s+!else\n([\s\S]*?)\n\s+!endif/.exec(template);
  assert.ok(own, "electron-builder's uninstaller no longer takes customRemoveFiles: read templates/nsis/uninstaller.nsh");
  for (const name of ["un.atomicRMDir", "un.restoreFiles"]) {
    assert.match(template, new RegExp(`^Function ${name.replace(".", "[.]")}$`, "m"), `${name} is gone from the template`);
  }
  // It still moves the files one by one into its temporary directory.
  assert.match(own[1], /Call un\.atomicRMDir/);

  const include = fs.readFileSync(INCLUDE, "utf8").replace(/\r\n/g, "\n");
  const code = (text) => lines(text).filter((line) => !line.startsWith("#"));
  const ours = code(macro(include, "customRemoveFiles"));
  // Theirs: `${if} ${isUpdated} <move file by file> ${endif} <delete>`.
  const theirs = code(own[1]);
  const updated = theirs.indexOf("${if} ${isUpdated}");
  const closing = theirs.lastIndexOf("${endif}");
  assert.ok(updated === 0 && closing > 0, "electron-builder changed the shape of its removal");
  const moveFileByFile = theirs.slice(updated + 1, closing);
  const thenDelete = theirs.slice(closing + 1);
  const start = ours.indexOf('CreateDirectory "$PLUGINSDIR\\old-install"');
  assert.ok(start > 0, "customRemoveFiles no longer falls back to electron-builder's removal");
  assert.deepEqual(
    ours.slice(start, start + moveFileByFile.length + thenDelete.length),
    [...moveFileByFile, ...thenDelete],
    "electron-builder changed how its uninstaller removes the old files: carry the change over to " +
      "customRemoveFiles in resources/installer.nsh",
  );
  assert.deepEqual(ours.slice(-2 - thenDelete.length, -2), thenDelete, "the plain uninstall is no longer the template's");
});

test("the Windows installer stays one that an update can download in parts", () => {
  // Solid, it would be 83 MB smaller (measured: 472 MB against 555 MB) and
  // have no block map: every update would be all of it. `useZip` only takes
  // effect on such an installer.
  for (const variant of ["", "voice"]) {
    const { nsis } = configured({ AUTOGPT_DESKTOP_VARIANT: variant });
    assert.notEqual(nsis.differentialPackage, false);
    assert.equal(nsis.useZip, undefined);
  }
});
