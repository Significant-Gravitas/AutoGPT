"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const { NORMAL } = require("./identity");

// A file in the data directory that names the install the directory belongs
// to (its `dirName`).
const INSTALL_STAMP = "autogpt-install.json";
// What a data directory holds once the runtime has run on it.
const USED = ["config", "postgres"];

// Databases and agent workspaces grow large, so they live in the per-machine
// location of each OS rather than a roaming profile. `install.dirName` is the
// install's own folder there (identity.js): a variant never shares one with
// the normal app or with another variant.
/** @param {{ variant: string, dirName: string }} [install] an identity (identity.js); the installed-app tests are type-checked against this */
function defaultDataDir(platform = process.platform, env = process.env, install = NORMAL) {
  if (env.AUTOGPT_DESKTOP_DATA_DIR) return movedDataDir(env.AUTOGPT_DESKTOP_DATA_DIR, install);
  const { dirName } = install;
  const home = os.homedir();
  if (platform === "win32") {
    return path.join(env.LOCALAPPDATA || path.join(home, "AppData", "Local"), dirName);
  }
  if (platform === "darwin") {
    return path.join(home, "Library", "Application Support", dirName);
  }
  return path.join(env.XDG_DATA_HOME || path.join(home, ".local", "share"), dirName);
}

// AUTOGPT_DESKTOP_DATA_DIR names the normal app's data directory. The
// variable is one per user or per shell, not one per install, so a variant
// that took it as it is would open the normal app's database: it takes the
// folder next to it, `<directory>-<slug>`.
function movedDataDir(directory, install) {
  if (!install.variant) return directory;
  const normal = path.resolve(directory);
  if (normal === path.parse(normal).root) return path.join(normal, install.dirName);
  return `${normal}-${install.variant}`;
}

// Whose data directory this is, after taking it for `install` if it is
// nobody's yet: the `dirName` of the install it belongs to. The caller must
// not touch a directory that is another install's; two installs can be
// pointed at one directory by a link, or by a data directory given to one
// of them that happens to be the other's.
//
// A directory that was used before installs left their name in it is the
// normal app's: there was nothing else.
function claimDataDir(dataDir, install) {
  const stamp = path.join(dataDir, INSTALL_STAMP);
  const owner = readStamp(stamp);
  if (owner) return owner;
  if (install.variant && USED.some((name) => fs.existsSync(path.join(dataDir, name)))) return NORMAL.dirName;
  fs.mkdirSync(dataDir, { recursive: true });
  return writeStamp(stamp, install.dirName);
}

function readStamp(stamp) {
  try {
    const owner = JSON.parse(fs.readFileSync(stamp, "utf8")).install;
    return typeof owner === "string" && owner !== "" ? owner : null;
  } catch {
    return null;
  }
}

// Two installs started at the same moment must not both believe the
// directory is theirs: the file appears whole and only if it is not there
// (a hard link), and whoever loses reads the winner's name.
function writeStamp(stamp, dirName) {
  const temporary = `${stamp}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify({ install: dirName })}\n`);
  try {
    fs.linkSync(temporary, stamp);
  } catch (error) {
    if (error.code !== "EEXIST") {
      // A file system without hard links (exFAT on an external disk).
      return writeStampInPlace(stamp, dirName);
    }
    // A file that names nobody is damage, not another install.
    const owner = readStamp(stamp);
    if (owner) return owner;
    fs.renameSync(temporary, stamp);
    return dirName;
  } finally {
    fs.rmSync(temporary, { force: true });
  }
  return dirName;
}

function writeStampInPlace(stamp, dirName) {
  try {
    fs.writeFileSync(stamp, `${JSON.stringify({ install: dirName })}\n`, { flag: "wx" });
    return dirName;
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
    return readStamp(stamp) || dirName;
  }
}

// A packaged app carries its runtime under resources/runtime. During
// development, point AUTOGPT_DESKTOP_RUNTIME at a staged runtime directory.
function runtimeDir({ isPackaged, resourcesPath, env = process.env }) {
  if (env.AUTOGPT_DESKTOP_RUNTIME) return env.AUTOGPT_DESKTOP_RUNTIME;
  if (isPackaged) return path.join(resourcesPath, "runtime");
  return path.join(__dirname, "..", "build", "runtime");
}

// runtime/manifest.json names the process the shell starts. Paths in it are
// relative to the runtime directory, so the manifest is identical wherever
// the app is installed.
function readRuntimeManifest(dir, platform = process.platform) {
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, "manifest.json"), "utf8"));
  const entry = manifest[platform] || manifest.default;
  if (!entry || typeof entry.command !== "string") {
    throw new Error(`runtime manifest has no command for ${platform}`);
  }
  const resolve = (value) =>
    typeof value === "string" ? value.replaceAll("{runtime}", dir) : value;
  return {
    command: path.isAbsolute(entry.command) ? entry.command : path.join(dir, entry.command),
    args: (entry.args || []).map(resolve),
    env: Object.fromEntries(
      Object.entries(entry.env || {}).map(([key, value]) => [key, resolve(value)]),
    ),
  };
}

module.exports = { INSTALL_STAMP, claimDataDir, defaultDataDir, runtimeDir, readRuntimeManifest };
