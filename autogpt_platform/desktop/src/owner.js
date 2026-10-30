"use strict";

// The owner's corner of the menus, without Electron so it can be tested.
//
// The first account created on the machine is its owner and an admin, and
// registration then closes (runtime/autogpt_desktop/bootstrap.py). Sign-in
// is by email and password with no mail server behind it, so "forgot
// password" cannot send a link. The way back in is local: the shell writes
// the new password to a private one-shot file and restarts the runtime,
// which reads it, deletes it and stores the hash for the owner.

const fs = require("node:fs");
const path = require("node:path");

// frontend/src/lib/auth/password-policy.ts; the runtime enforces the same
// limits when it reads the file (bootstrap.PASSWORD_MIN_LENGTH).
const PASSWORD_MIN_LENGTH = 12;
const PASSWORD_MAX_LENGTH = 128;
// Everything Python's str.splitlines() ends a line at.
const LINE_BREAK = new RegExp(
  `[\\n\\v\\f\\r\\x1c-\\x1e\\x85${String.fromCharCode(0x2028, 0x2029)}]`,
);
// bootstrap.RESET_PASSWORD_FILE, inside the data directory's config folder.
const RESET_PASSWORD_FILE = "reset-password";

// Where a provider sends the user back after they connect an integration:
// what to register as the redirect (callback) URL of an OAuth app.
function oauthRedirectUrl(appUrl) {
  if (!appUrl) return null;
  return `${new URL(appUrl).origin}/auth/integrations/oauth_callback`;
}

// What is wrong with a new password, or null. The runtime reads one line
// and counts characters the way Python does, not UTF-16 units.
function passwordProblem(password, confirmation) {
  const length = typeof password === "string" ? Array.from(password).length : 0;
  if (length < PASSWORD_MIN_LENGTH) {
    return `Use at least ${PASSWORD_MIN_LENGTH} characters.`;
  }
  if (length > PASSWORD_MAX_LENGTH) {
    return `Use at most ${PASSWORD_MAX_LENGTH} characters.`;
  }
  if (LINE_BREAK.test(password)) return "A password cannot contain a line break.";
  if (password !== confirmation) return "The two passwords do not match.";
  return null;
}

function resetPasswordFile(dataDir) {
  return path.join(dataDir, "config", RESET_PASSWORD_FILE);
}

// Private to the user from the moment it exists, and never half-written:
// the runtime may be reading the directory while this runs.
function writePasswordReset(dataDir, password) {
  const target = resetPasswordFile(dataDir);
  const temporary = `${target}.${process.pid}.tmp`;
  fs.mkdirSync(path.dirname(target), { recursive: true, mode: 0o700 });
  fs.rmSync(temporary, { force: true });
  try {
    fs.writeFileSync(temporary, `${password}\n`, { mode: 0o600, flag: "wx" });
    fs.renameSync(temporary, target);
  } catch (error) {
    fs.rmSync(temporary, { force: true });
    throw error;
  }
  return target;
}

// The same entries go in the tray menu and in the application menu: a tray
// icon can be hidden, or missing altogether.
function ownerMenuItems({ appUrl, copyText, resetPassword }) {
  const redirectUrl = oauthRedirectUrl(appUrl);
  return [
    {
      label: "Copy OAuth redirect URL",
      enabled: Boolean(redirectUrl),
      click: () => redirectUrl && copyText(redirectUrl),
    },
    { label: "Reset owner password…", click: () => resetPassword() },
  ];
}

// macOS routes Cmd+C/V/A/Q through the application menu, so it needs the
// standard ones. Elsewhere a menu bar is clutter and the windows keep it
// hidden until Alt is pressed; it exists because GNOME shows no tray icon
// without an extension, and the owner's actions need a way in that is always
// there.
function applicationMenuTemplate(platform, ownerItems) {
  if (platform !== "darwin") return [{ label: "&Account", submenu: ownerItems }];
  return [
    { role: "appMenu" },
    { role: "editMenu" },
    { role: "viewMenu" },
    { label: "Account", submenu: ownerItems },
    { role: "windowMenu" },
  ];
}

module.exports = {
  PASSWORD_MIN_LENGTH,
  PASSWORD_MAX_LENGTH,
  RESET_PASSWORD_FILE,
  applicationMenuTemplate,
  oauthRedirectUrl,
  ownerMenuItems,
  passwordProblem,
  resetPasswordFile,
  writePasswordReset,
};
