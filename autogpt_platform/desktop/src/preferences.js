"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const DEFAULT_PREFERENCES = Object.freeze({
  version: 1,
  usageStatistics: false,
  crashReports: false,
  launchAtLogin: false,
  keepRunningInBackground: false,
});
const BOOLEAN_KEYS = Object.keys(DEFAULT_PREFERENCES).filter((key) => key !== "version");

function validatePatch(patch) {
  if (!patch || typeof patch !== "object" || ![Object.prototype, null].includes(Object.getPrototypeOf(patch))) {
    throw new TypeError("Desktop preferences must be an object.");
  }
  for (const key of Reflect.ownKeys(patch)) {
    if (key === "version" && patch[key] === 1) continue;
    if (!BOOLEAN_KEYS.includes(key) || typeof patch[key] !== "boolean") {
      throw new TypeError("Desktop preferences contain an unknown key or invalid value.");
    }
  }
}

function normalizePreferences(value) {
  try {
    validatePatch(value);
    if (value.version !== 1 || !BOOLEAN_KEYS.every((key) => Object.hasOwn(value, key))) {
      return DEFAULT_PREFERENCES;
    }
    return Object.freeze({ ...DEFAULT_PREFERENCES, ...value });
  } catch {
    return DEFAULT_PREFERENCES;
  }
}

function preferencesPath(dataDir) {
  return path.join(dataDir, "config", "desktop-preferences.json");
}

function loadPreferences(dataDir) {
  try {
    const file = preferencesPath(dataDir);
    const directory = fs.lstatSync(path.dirname(file));
    const stat = fs.lstatSync(file);
    if (!directory.isDirectory() || directory.isSymbolicLink() || !stat.isFile() || stat.isSymbolicLink() || stat.size > 4096) {
      return DEFAULT_PREFERENCES;
    }
    return normalizePreferences(JSON.parse(fs.readFileSync(file, "utf8")));
  } catch {
    return DEFAULT_PREFERENCES;
  }
}

function savePreferences(dataDir, patch) {
  validatePatch(patch);
  const next = Object.freeze({ ...loadPreferences(dataDir), ...patch, version: 1 });
  const file = preferencesPath(dataDir);
  const directory = path.dirname(file);
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
  const directoryStat = fs.lstatSync(directory);
  if (!directoryStat.isDirectory() || directoryStat.isSymbolicLink()) {
    throw new Error("Desktop preferences require a regular configuration directory.");
  }
  try {
    const existing = fs.lstatSync(file);
    if (!existing.isFile() || existing.isSymbolicLink()) {
      throw new Error("Desktop preferences require a regular file.");
    }
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  const temporary = path.join(directory, `.desktop-preferences-${crypto.randomUUID()}.tmp`);
  try {
    const descriptor = fs.openSync(temporary, "wx", 0o600);
    try {
      fs.writeFileSync(descriptor, `${JSON.stringify(next, null, 2)}\n`, "utf8");
      fs.fsyncSync(descriptor);
    } finally {
      fs.closeSync(descriptor);
    }
    fs.renameSync(temporary, file);
    if (process.platform !== "win32") {
      const descriptor = fs.openSync(directory, "r");
      try {
        fs.fsyncSync(descriptor);
      } finally {
        fs.closeSync(descriptor);
      }
    }
  } finally {
    fs.rmSync(temporary, { force: true });
  }
  return next;
}

module.exports = { DEFAULT_PREFERENCES, loadPreferences, normalizePreferences, savePreferences };
