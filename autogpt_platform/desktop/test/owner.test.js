"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { test } = require("node:test");

const {
  PASSWORD_MAX_LENGTH,
  PASSWORD_MIN_LENGTH,
  RESET_PASSWORD_FILE,
  applicationMenuTemplate,
  oauthRedirectUrl,
  ownerMenuItems,
  passwordProblem,
  resetPasswordFile,
  writePasswordReset,
} = require("../src/owner");

const APP = "http://127.0.0.1:18473";
const GOOD = "correct horse battery";
const PLATFORM = path.join(__dirname, "..", "..");

function temporaryDataDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "autogpt owner "));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

test("the OAuth redirect URL is the app's origin plus the callback path", () => {
  assert.equal(oauthRedirectUrl(APP), `${APP}/auth/integrations/oauth_callback`);
  assert.equal(oauthRedirectUrl(`${APP}/`), `${APP}/auth/integrations/oauth_callback`);
  assert.equal(oauthRedirectUrl(null), null);
});

test("the redirect path is still the one the backend hands to providers", () => {
  const router = fs.readFileSync(
    path.join(PLATFORM, "backend", "backend", "api", "features", "integrations", "router.py"),
    "utf8",
  );
  assert.ok(
    router.includes("/auth/integrations/oauth_callback"),
    "The backend no longer uses /auth/integrations/oauth_callback: update oauthRedirectUrl in desktop/src/owner.js and the README.",
  );
});

test("a new password must be long enough, typed twice and one line", () => {
  assert.equal(passwordProblem(GOOD, GOOD), null);
  assert.match(passwordProblem("short", "short"), /at least 12/);
  assert.match(passwordProblem(undefined, undefined), /at least 12/);
  assert.match(passwordProblem(GOOD, `${GOOD}!`), /do not match/);
  assert.match(passwordProblem(`${GOOD}\nmore`, `${GOOD}\nmore`), /line break/);
  const separated = `${GOOD}${String.fromCharCode(0x2028)}more`; // LINE SEPARATOR
  assert.match(passwordProblem(separated, separated), /line break/);
  const long = "x".repeat(PASSWORD_MAX_LENGTH + 1);
  assert.match(passwordProblem(long, long), /at most 128/);
  assert.equal(passwordProblem(long.slice(1), long.slice(1)), null);
});

test("length is counted in characters, as the runtime counts it", () => {
  const six = "🔑".repeat(6); // twelve UTF-16 units, six characters
  assert.equal(six.length, 12);
  assert.match(passwordProblem(six, six), /at least 12/);
  assert.equal(passwordProblem(six + six, six + six), null);
});

test("the password rules are the frontend's and the runtime's", () => {
  const policy = fs.readFileSync(
    path.join(PLATFORM, "frontend", "src", "lib", "auth", "password-policy.ts"),
    "utf8",
  );
  const runtime = fs.readFileSync(
    path.join(__dirname, "..", "runtime", "autogpt_desktop", "bootstrap.py"),
    "utf8",
  );
  const update = "Update PASSWORD_* and RESET_PASSWORD_FILE in desktop/src/owner.js.";
  assert.ok(policy.includes(`AUTH_PASSWORD_MIN_LENGTH = ${PASSWORD_MIN_LENGTH};`), update);
  const lines = runtime.split(/\r?\n/);
  assert.ok(lines.includes(`PASSWORD_MIN_LENGTH = ${PASSWORD_MIN_LENGTH}`), update);
  assert.ok(lines.includes(`PASSWORD_MAX_LENGTH = ${PASSWORD_MAX_LENGTH}`), update);
  assert.ok(lines.includes(`RESET_PASSWORD_FILE = "${RESET_PASSWORD_FILE}"`), update);
});

test("the reset file is one private line where the runtime looks for it", (t) => {
  const dataDir = temporaryDataDir(t);

  const written = writePasswordReset(dataDir, GOOD);

  assert.equal(written, path.join(dataDir, "config", "reset-password"));
  assert.equal(written, resetPasswordFile(dataDir));
  assert.equal(fs.readFileSync(written, "utf8"), `${GOOD}\n`);
  assert.deepEqual(fs.readdirSync(path.dirname(written)), ["reset-password"]);
  if (process.platform !== "win32") {
    assert.equal(fs.statSync(written).mode & 0o777, 0o600);
    assert.equal(fs.statSync(path.dirname(written)).mode & 0o777, 0o700);
  }
});

test("a second reset replaces the first, and a crashed one leaves no blocker", (t) => {
  const dataDir = temporaryDataDir(t);
  const stale = `${resetPasswordFile(dataDir)}.${process.pid}.tmp`;
  fs.mkdirSync(path.dirname(stale), { recursive: true });
  fs.writeFileSync(stale, "left by a crash");

  writePasswordReset(dataDir, GOOD);
  writePasswordReset(dataDir, `${GOOD} again`);

  assert.equal(fs.readFileSync(resetPasswordFile(dataDir), "utf8"), `${GOOD} again\n`);
  assert.deepEqual(fs.readdirSync(path.dirname(stale)), ["reset-password"]);
});

test("a reset that cannot be written leaves no half-written file", (t) => {
  const dataDir = temporaryDataDir(t);
  // A directory where the file should go makes the final rename fail.
  fs.mkdirSync(path.join(resetPasswordFile(dataDir), "in the way"), { recursive: true });

  assert.throws(() => writePasswordReset(dataDir, GOOD));

  assert.deepEqual(fs.readdirSync(path.join(dataDir, "config")), ["reset-password"]);
});

test("the owner menu copies the real redirect URL once the app is ready", () => {
  const copied = [];
  let resets = 0;
  const handlers = { copyText: (text) => copied.push(text), resetPassword: () => resets++ };

  const [copyBefore, resetBefore] = ownerMenuItems({ appUrl: null, ...handlers });
  assert.equal(copyBefore.enabled, false);
  copyBefore.click();
  assert.deepEqual(copied, []);
  // Resetting must work while the app is still starting, or failed to.
  assert.notEqual(resetBefore.enabled, false);

  const [copy, reset] = ownerMenuItems({ appUrl: "http://127.0.0.1:20123", ...handlers });
  assert.equal(copy.label, "Copy OAuth redirect URL");
  assert.equal(copy.enabled, true);
  copy.click();
  assert.deepEqual(copied, ["http://127.0.0.1:20123/auth/integrations/oauth_callback"]);
  assert.equal(reset.label, "Reset owner password…");
  reset.click();
  assert.equal(resets, 1);
});

test("the owner's actions are in the application menu on every system", () => {
  const items = ownerMenuItems({ appUrl: APP, copyText() {}, resetPassword() {} });
  for (const platform of ["win32", "linux", "darwin"]) {
    const account = applicationMenuTemplate(platform, items).find((entry) => entry.submenu);
    assert.equal(account.label.replace("&", ""), "Account", platform);
    assert.deepEqual(
      account.submenu.map((item) => item.label),
      ["Copy OAuth redirect URL", "Reset owner password\u2026"],
      platform,
    );
  }
});

test("macOS keeps its standard menus; elsewhere the hidden menu holds the account only", () => {
  const roles = (platform) => applicationMenuTemplate(platform, []).map((entry) => entry.role);
  assert.deepEqual(roles("darwin"), ["appMenu", "editMenu", "viewMenu", undefined, "windowMenu"]);
  assert.deepEqual(roles("linux"), [undefined]);
  assert.deepEqual(roles("win32"), [undefined]);
});

test("the windows keep that menu out of sight until Alt is pressed", () => {
  const main = fs.readFileSync(path.join(__dirname, "..", "src", "main.js"), "utf8");
  const windows = main.split("new BrowserWindow(").slice(1);
  assert.equal(windows.length, 2);
  for (const options of windows) {
    assert.ok(options.slice(0, options.indexOf("webPreferences")).includes("autoHideMenuBar: true"));
  }
});
