"use strict";

const PREFERENCES = new Set(["usageStatistics", "crashReports", "launchAtLogin", "keepRunningInBackground"]);

function backgroundOnClose({ keepRunningInBackground, quitting, sessionEnding, runtimeExited }) {
  return keepRunningInBackground === true && !quitting && !sessionEnding && !runtimeExited;
}

function loginAvailability({ isPackaged, platform, customProfile }) {
  if (!isPackaged) return { supported: false, reason: "Install AutoGPT to enable launch at login." };
  if (customProfile) return { supported: false, reason: "Launch at login is unavailable for a custom data or browser profile." };
  if (!["win32", "darwin"].includes(platform)) {
    return { supported: false, reason: "Use your desktop's startup applications settings on Linux." };
  }
  return { supported: true, reason: "Open AutoGPT when you sign in to this computer." };
}

function loginOptions(enabled, platform, executable, name) {
  return platform === "win32"
    ? { openAtLogin: enabled, enabled, path: executable, args: [], name }
    : { openAtLogin: enabled };
}

async function applyPreferences({ current, patch, supported, save, setLogin, getLogin = () => current.launchAtLogin }) {
  if (!patch || typeof patch !== "object" || Array.isArray(patch) ||
      Object.entries(patch).some(([key, value]) => !PREFERENCES.has(key) || typeof value !== "boolean")) {
    throw new Error("Choose true or false for each desktop preference.");
  }
  const changedLogin = Object.hasOwn(patch, "launchAtLogin") && patch.launchAtLogin !== current.launchAtLogin;
  if (changedLogin && !supported) throw new Error("Launch at login is unavailable for this installation.");
  const applyLogin = supported && Object.hasOwn(patch, "launchAtLogin");
  const previousLogin = applyLogin ? await getLogin() : false;
  if (applyLogin) await setLogin(patch.launchAtLogin);
  try {
    return await save(patch);
  } catch (error) {
    if (applyLogin) {
      try { await setLogin(previousLogin); }
      catch { throw new Error("Preferences could not be saved. Check this computer's startup applications settings."); }
    }
    throw error;
  }
}

module.exports = { backgroundOnClose, loginAvailability, loginOptions, applyPreferences };
