"use strict";

// Who an install is, without Electron so it can be tested and so that the
// build, the shell and the installed-app tests all read the one definition.
//
// The fork builds the normal app and any number of variants: experiment
// branches (`variant/<slug>`) built with AUTOGPT_DESKTOP_VARIANT=<slug>. A
// variant installs next to the normal app and next to other variants, and
// shares nothing with them. Everything that tells two installs apart is
// derived here from the slug, and from nothing else:
//
//                      normal app               variant <slug>
//   product name       AutoGPT                  AutoGPT (<slug>)
//   app id             co.agpt.autogpt.desktop  co.agpt.autogpt.desktop.<slug>
//   package name       autogpt                  autogpt-<slug>
//   executable         AutoGPT                  autogpt-<slug>
//   data and profile   AutoGPT                  AutoGPT-<slug>
//   installers         AutoGPT-...              AutoGPT-<slug>-...
//   public port        18473                    20000-29999, from the slug
//   release tags       desktop-v<version>       desktop-<slug>-v<version>
//
// The normal app's column is what was shipped before there were variants.
// It must never change: an installed app would lose its data, its sign-ins
// and its updates. test/variants.test.js holds it in place.
//
// Run as a program (see `run` at the end), this prints the names a workflow
// needs, one `key=value` per line, for the variant in AUTOGPT_DESKTOP_VARIANT.

const crypto = require("node:crypto");
const path = require("node:path");

const VARIANT_VARIABLE = "AUTOGPT_DESKTOP_VARIANT";
// Lower-case letters, digits and hyphens, 1 to 24 characters, not starting
// or ending with a hyphen: the slug becomes part of a bundle identifier, a
// Debian package name, a file name, a git tag and a branch name.
const SLUG = /^[a-z0-9](?:[a-z0-9-]{0,22}[a-z0-9])?$/;
// electron-updater keeps what it downloads in `<package name>-updater`, in
// the folder that on Windows and macOS also holds the data directories.
// Those file systems ignore case, so a variant `updater` would keep its
// database in the normal app's download cache, and `x-updater` in the cache
// of the variant `x`.
const TAKEN = /(?:^|-)updater$/;

// runtime/autogpt_desktop/ports.py: PREFERRED["public"], and a part of
// PORT_RANGE that the normal app's port is not in. The runtime's tests read
// these three lines.
const NORMAL_PUBLIC_PORT = 18473;
const VARIANT_PORT_FIRST = 20000;
const VARIANT_PORT_COUNT = 10000;

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
  branch: "desktop",
});

function identity(variant = "") {
  if (variant === "") return NORMAL;
  if (typeof variant !== "string" || !SLUG.test(variant)) {
    throw new Error(
      `${VARIANT_VARIABLE} must be 1 to 24 lower-case letters, digits or hyphens, ` +
        `not starting or ending with a hyphen; got ${JSON.stringify(variant)}`,
    );
  }
  if (TAKEN.test(variant)) {
    throw new Error(
      `${VARIANT_VARIABLE} must not be "updater" or end in "-updater": its data directory ` +
        `would be another install's download cache; got ${JSON.stringify(variant)}`,
    );
  }
  return Object.freeze({
    variant,
    productName: `AutoGPT (${variant})`,
    appId: `${NORMAL.appId}.${variant}`,
    packageName: `autogpt-${variant}`,
    // Not the product name: it has a space and brackets in it, which the
    // AppImage launcher refuses and which would end up in every path of the
    // macOS bundle.
    executableName: `autogpt-${variant}`,
    productFilename: `autogpt-${variant}`,
    dirName: `AutoGPT-${variant}`,
    artifactBase: `AutoGPT-${variant}`,
    publicPort: variantPort(variant),
    tagPrefix: `desktop-${variant}-v`,
    branch: `variant/${variant}`,
  });
}

// Stable for a slug on every machine, so that a variant's address, like the
// normal app's, can be written down once. Two slugs can land on one port
// (the range has 10000); the runtime then moves whichever starts second.
function variantPort(variant) {
  const digest = crypto.createHash("sha256").update(variant, "utf8").digest();
  return VARIANT_PORT_FIRST + (digest.readUInt32BE(0) % VARIANT_PORT_COUNT);
}

// Which variant this process is. An installed app is what it was built as
// (electron-builder.config.js writes the slug into its package.json) and the
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

function workflowOutputs(id) {
  return {
    variant: id.variant,
    product_name: id.productName,
    product_filename: id.productFilename,
    package_name: id.packageName,
    dir_name: id.dirName,
    artifact_base: id.artifactBase,
    tag_prefix: id.tagPrefix,
    branch: id.branch,
  };
}

// node src/identity.js                           the names, for a workflow
// node src/identity.js owns <version> <file>...  fails unless an installed
//                                                app would take every file
//                                                as its own update
function run([command, version, ...files], env) {
  const id = identity(env[VARIANT_VARIABLE] || "");
  if (command !== "owns") {
    return { code: 0, lines: Object.entries(workflowOutputs(id)).map(([key, value]) => `${key}=${value}`) };
  }
  const foreign = files.filter((file) => !isOwnUpdateFile(id, version, file));
  const lines = foreign.map(
    (file) => `${file} is not an installer of ${id.productName} ${version}: installed apps would refuse the update`,
  );
  return { code: foreign.length > 0 ? 1 : 0, lines };
}

if (require.main === module) {
  const { code, lines } = run(process.argv.slice(2), process.env);
  for (const line of lines) process.stdout.write(`${line}\n`);
  process.exitCode = code;
}

module.exports = {
  NORMAL,
  SLUG,
  VARIANT_VARIABLE,
  artifactNames,
  claim,
  identity,
  isOwnUninstallName,
  isOwnUpdateFile,
  releaseTag,
  run,
  runtimeEnvironment,
  variantOf,
  windowTitle,
  workflowOutputs,
};
