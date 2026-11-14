"use strict";

// Stable identities for the standard app and the experimental CI distribution.
// Keep their existing profile paths, installer names and public ports so an
// upgrade continues to use the same data and browser sessions.

const path = require("node:path");

const VARIANT_VARIABLE = "AUTOGPT_DESKTOP_VARIANT";
const NORMAL_PUBLIC_PORT = 18473;
const EXPERIMENTAL_PUBLIC_PORT = 25500;

const NORMAL = Object.freeze({
  variant: "",
  productName: "AutoGPT",
  appId: "co.agpt.autogpt.desktop",
  packageName: "autogpt",
  // electron-builder names the executable after the product unless told
  // otherwise, and for the normal app it is not told.
  executableName: null,
  productFilename: "AutoGPT",
  dirName: "AutoGPT",
  artifactBase: "AutoGPT",
  publicPort: NORMAL_PUBLIC_PORT,
  tagPrefix: "desktop-v",
});

const EXPERIMENTAL = Object.freeze({
  variant: "experimental",
  productName: "AutoGPT (experimental)",
  appId: "co.agpt.autogpt.desktop.experimental",
  packageName: "autogpt-experimental",
  executableName: "autogpt-experimental",
  productFilename: "autogpt-experimental",
  dirName: "AutoGPT-experimental",
  artifactBase: "AutoGPT-experimental",
  publicPort: EXPERIMENTAL_PUBLIC_PORT,
  tagPrefix: "desktop-experimental-v",
});

function identity(variant = "") {
  if (variant === "") return NORMAL;
  if (variant === "experimental") return EXPERIMENTAL;
  throw new Error(`${VARIANT_VARIABLE} must be empty or "experimental"; got ${JSON.stringify(variant)}`);
}

// Which variant this process is. An installed app is what it was built as
// (electron-builder.config.js records it in package.json) and the
// environment cannot make it something else: a variable left set in a shell
// must not point the normal app at a variant's data, or the reverse.
function variantOf({ isPackaged, manifest = {}, env = {} }) {
  const built = manifest.autogptDesktop?.variant;
  if (isPackaged || built) return built || "";
  return env[VARIANT_VARIABLE] || "";
}

// electron-builder's artifactName patterns. The AppImage has no version in
// its name: it updates itself by replacing its own file, and a launcher that
// points at it must keep working.
function artifactNames(id) {
  return {
    nsis: `${id.artifactBase}-Setup-\${version}-\${arch}.\${ext}`,
    mac: `${id.artifactBase}-\${version}-\${arch}.\${ext}`,
    appImage: `${id.artifactBase}-\${arch}.\${ext}`,
    deb: `${id.artifactBase}-\${version}-\${arch}.\${ext}`,
  };
}

// Whether a file named in a release's update metadata is this app's own
// installer for that version (the patterns above, filled in). A plain name
// only: an address would let the metadata send the app anywhere.
function isOwnUpdateFile(id, version, name) {
  if (typeof name !== "string" || typeof version !== "string" || version === "") return false;
  const base = escapeRegExp(id.artifactBase);
  const v = escapeRegExp(version);
  const arch = "[A-Za-z0-9_]+";
  const own = new RegExp(
    `^${base}-(?:Setup-${v}-${arch}[.]exe|${v}-${arch}[.](?:dmg|zip|deb)|${arch}[.]AppImage)$`,
  );
  return own.test(name);
}

// Whether a DisplayName among Windows' uninstall entries is this app's. The
// installer writes "<product name> <version>" (electron-builder's NSIS
// target), and the normal app's name is the start of every variant's.
function isOwnUninstallName(id, displayName) {
  return typeof displayName === "string" && new RegExp(`^${escapeRegExp(id.productName)} [0-9]`).test(displayName);
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function releaseTag(id, version) {
  return `${id.tagPrefix}${version}`;
}

// What the shell adds to the runtime's environment. The runtime knows
// nothing about variants; it is told the port to prefer and the install's
// name, under which it keeps what installs would otherwise share: its few
// files outside the data directory, and its cookies in a browser.
function runtimeEnvironment(id) {
  return {
    AUTOGPT_DESKTOP_PUBLIC_PORT: String(id.publicPort),
    AUTOGPT_DESKTOP_INSTALL_NAME: id.dirName,
  };
}

// Before the single-instance lock is asked for, and before the app is ready:
// Electron keys the lock, the profile, its caches and its crash reports on
// the profile directory. For the normal app this names what Electron picks
// anyway. A profile given on the command line (the installed-app tests) wins.
function claim(app, id) {
  app.setName(id.productName);
  if (app.commandLine.hasSwitch("user-data-dir")) return;
  app.setPath("userData", path.join(app.getPath("appData"), id.dirName));
}

// The title of a window showing a page titled `pageTitle`, or null to leave
// it alone. The platform's pages name themselves the same in every install,
// so a variant's windows say which one they are.
function windowTitle(id, pageTitle) {
  if (!id.variant) return null;
  if (!pageTitle || pageTitle === NORMAL.productName) return id.productName;
  return pageTitle.includes(id.productName) ? pageTitle : `${pageTitle} - ${id.productName}`;
}

module.exports = {
  NORMAL,
  EXPERIMENTAL,
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
};
