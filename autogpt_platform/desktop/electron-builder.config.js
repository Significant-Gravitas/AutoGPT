"use strict";

// How the installers are built:
//
//   npx electron-builder --config electron-builder.config.js --publish never
//
// One configuration for every kind of build. What differs between a
// developer's build and a signed release comes from the environment, so that
// nothing in the repository is edited to cut a release:
//
//   AUTOGPT_DESKTOP_VERSION    the app's version; package.json's
//                              0.0.0-dev.0 otherwise (a development build,
//                              which never looks for updates)
//   AUTOGPT_DESKTOP_OUTPUT     where the installers are written; dist
//   AUTOGPT_DESKTOP_VARIANT    the slug of a variant, such as "voice": an app
//                              that installs next to the normal one and
//                              shares nothing with it. Empty for the normal
//                              app. src/identity.js derives every name here
//                              from it; README.md, "Variants".
//   AUTOGPT_DESKTOP_MAC_SIGN   "developer-id": sign with the Developer ID
//                              certificate in CSC_LINK (or the keychain),
//                              hardened runtime, notarize. Anything else:
//                              ad-hoc signature, no hardened runtime.
//   AUTOGPT_DESKTOP_WIN_SIGN   "azure": Azure Trusted Signing, described by
//                              AZURE_SIGN_ENDPOINT, AZURE_SIGN_ACCOUNT,
//                              AZURE_SIGN_PROFILE and AZURE_SIGN_PUBLISHER.
//                              "pfx": the certificate in WIN_CSC_LINK.
//                              Anything else: unsigned.
//
// The certificates and passwords themselves are read by electron-builder
// from its own variables (CSC_LINK, CSC_KEY_PASSWORD, APPLE_API_KEY, ...,
// WIN_CSC_LINK, AZURE_CLIENT_SECRET, ...). They never pass through this file.

const path = require("node:path");

// To sign a macOS app, @electron/osx-sign opens every file in it at once to
// see which are programs, and the bundle has tens of thousands. Past the
// system's limit on open files that is an error, not a wait: it failed on
// GitHub's macOS machines, and passed on a Mac whose limit happened to be
// higher than the number of files. With this, an open in this process waits
// for a free slot instead. It must run before that library is loaded, which
// electron-builder does only when it signs. Where nothing is installed
// (the unit tests read this file bare) there is nothing to package either.
try {
  require("graceful-fs").gracefulify(require("node:fs"));
} catch (error) {
  if (error.code !== "MODULE_NOT_FOUND") throw error;
}

const { artifactNames, identity, releaseTag } = require("./src/identity");
const { releaseFeed } = require("./src/updater");
const { beforePack, afterPack } = require("./build/runtime_check");
const {
  requireNotarizedApp,
  signMacApp,
  vendorSignedPaths,
  SHELL_ENTITLEMENTS,
  HELPER_ENTITLEMENTS,
} = require("./build/mac_sign");

// Semantic versioning to the letter: no leading zeros (1.2.03, 1.2.3-rc.01).
// electron-builder would take them, and electron-updater then refuses to
// start in the app that has such a version and to read it from latest.yml.
const NUMBER = "(0|[1-9][0-9]*)";
const IDENTIFIER = "(0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)";
const SEMVER = new RegExp(`^${NUMBER}[.]${NUMBER}[.]${NUMBER}(-${IDENTIFIER}([.]${IDENTIFIER})*)?$`);
const AZURE_SETTINGS = {
  endpoint: "AZURE_SIGN_ENDPOINT",
  codeSigningAccountName: "AZURE_SIGN_ACCOUNT",
  certificateProfileName: "AZURE_SIGN_PROFILE",
  publisherName: "AZURE_SIGN_PUBLISHER",
};

const env = process.env;
const app = identity(env.AUTOGPT_DESKTOP_VARIANT || "");
const names = artifactNames(app);
const macDeveloperId = env.AUTOGPT_DESKTOP_MAC_SIGN === "developer-id";
const windowsSigning = ["azure", "pfx"].includes(env.AUTOGPT_DESKTOP_WIN_SIGN)
  ? env.AUTOGPT_DESKTOP_WIN_SIGN
  : null;

function version() {
  const value = env.AUTOGPT_DESKTOP_VERSION;
  if (!value) return {};
  if (!SEMVER.test(value)) {
    throw new Error(`AUTOGPT_DESKTOP_VERSION must look like 1.2.3 or 1.2.3-rc.1, not "${value}"`);
  }
  return { version: value };
}

// What the packed app's package.json says besides what the file in the
// repository says. A variant is another package under another name, and
// src/identity.js variantOf reads the slug back in the installed app.
function metadata() {
  // Read by src/updater.js: macOS replaces an app in place only when the
  // old and the new one are signed by the same Developer ID.
  const built = { macDeveloperId };
  if (!app.variant) return { ...version(), autogptDesktop: built };
  return {
    ...version(),
    name: app.packageName,
    productName: app.productName,
    autogptDesktop: { ...built, variant: app.variant },
  };
}

// Where an installed app looks for updates (written to app-update.yml).
// Nothing is uploaded from here: the release workflow on `main` does that.
//
// The normal app follows the release GitHub marks as the repository's
// latest. A variant must never do that, so it is not given the means: the
// only address it is built with is the release it was built for, and
// src/updater.js moves it on to a newer release of the same variant once it
// has found one.
function publish() {
  if (!app.variant) return [{ provider: "github", owner: "ntindle", repo: "autogpt" }];
  const built = version().version || require("./package.json").version;
  return [releaseFeed(releaseTag(app, built))];
}

function mac() {
  const common = {
    category: "public.app-category.productivity",
    minimumSystemVersion: "15.0",
    artifactName: names.mac,
    gatekeeperAssess: false,
  };
  if (!macDeveloperId) {
    // No certificate. An unsigned bundle counts as damaged on Apple silicon;
    // an ad-hoc signature makes a download intact, though not trusted.
    return {
      ...common,
      target: [{ target: "dmg", arch: ["arm64"] }],
      identity: "-",
      hardenedRuntime: false,
      // The Claude Code CLI is Anthropic's signed program and ships
      // unmodified; an ad-hoc signature would replace Anthropic's.
      signIgnore: vendorSignedPaths(),
    };
  }
  return {
    ...common,
    // The zip is what an installed app updates itself from; the disk image
    // is for people. Neither is changed after it is built: the notarization
    // ticket is stapled to the app inside them, and a later change would
    // falsify the checksums in latest-mac.yml.
    target: [
      { target: "dmg", arch: ["arm64"] },
      { target: "zip", arch: ["arm64"] },
    ],
    hardenedRuntime: true,
    notarize: true,
    entitlements: SHELL_ENTITLEMENTS,
    entitlementsInherit: HELPER_ENTITLEMENTS,
    sign: signMacApp,
  };
}

function windows() {
  const common = { target: [{ target: "nsis", arch: ["x64"] }] };
  if (!windowsSigning) return common;
  return {
    ...common,
    // electron-builder signs every .exe it copies. The Claude Code CLI is
    // Anthropic's signed program and ships unmodified.
    signExts: ["!claude.exe"],
    ...(windowsSigning === "azure" ? { azureSignOptions: azureSignOptions() } : {}),
  };
}

function azureSignOptions() {
  const missing = Object.values(AZURE_SETTINGS).filter((name) => !env[name]);
  if (missing.length > 0) {
    throw new Error(`Azure Trusted Signing needs ${missing.join(", ")}`);
  }
  return Object.fromEntries(
    Object.entries(AZURE_SETTINGS).map(([option, name]) => [option, env[name]]),
  );
}

// On macOS electron-builder ignores forceCodeSigning when `mac.sign` is a
// function: with no Developer ID identity it packs an unsigned app and says
// nothing. This runs when it starts on the disk image and on the zip, and
// stops the build unless the app it packed is signed and notarized.
function signedAppOrNothing() {
  let checked = null;
  return (event) => {
    checked ||= requireNotarizedApp(path.join(path.dirname(event.file), "mac-arm64", `${app.productFilename}.app`));
    return checked;
  };
}

// A build that was asked to sign must not quietly come out unsigned. This
// is what makes electron-builder refuse on Windows.
function signsOnThisSystem() {
  if (process.platform === "darwin") return macDeveloperId;
  if (process.platform === "win32") return Boolean(windowsSigning);
  return false;
}

module.exports = {
  appId: app.appId,
  productName: app.productName,
  ...(app.executableName ? { executableName: app.executableName } : {}),
  directories: {
    output: env.AUTOGPT_DESKTOP_OUTPUT || "dist",
    buildResources: "resources",
  },
  files: ["src/**/*", "package.json"],
  extraResources: [{ from: "build/runtime", to: "runtime", filter: ["**/*"] }],
  beforePack,
  afterPack,
  asar: true,
  // The app has no native module, so there is nothing to compile for
  // Electron. Left on, electron-builder runs a rebuild of the dependencies
  // while it packages, which is also when it holds the certificates.
  npmRebuild: false,
  extraMetadata: metadata(),
  forceCodeSigning: signsOnThisSystem(),
  ...(macDeveloperId && process.platform === "darwin" ? { artifactBuildStarted: signedAppOrNothing() } : {}),
  publish: publish(),
  // The update files are always latest*.yml, also for 1.2.3-rc.1: there is
  // one channel.
  detectUpdateChannel: false,
  win: windows(),
  nsis: {
    oneClick: true,
    perMachine: false,
    deleteAppDataOnUninstall: false,
    artifactName: names.nsis,
    // `differentialPackage` is left at its default, on. Measured on the
    // Windows bundle: the installer is 555 MB with it and 472 MB without,
    // and without it there is no block map, so every update would be the
    // whole installer instead of the parts that changed (half a megabyte
    // for a change to the shell or to a backend module).
    //
    // Which running programs the installer closes: this install's, and not
    // those of an install whose folder name merely starts the same. And how
    // an update removes the version it replaces.
    include: "resources/installer.nsh",
  },
  mac: mac(),
  linux: {
    target: ["AppImage", "deb"],
    category: "Utility",
  },
  // No version in the AppImage's name (src/identity.js artifactNames): it
  // updates itself by replacing its own file, and a launcher that points at
  // it must keep working.
  appImage: { artifactName: names.appImage },
  deb: { artifactName: names.deb },
};
