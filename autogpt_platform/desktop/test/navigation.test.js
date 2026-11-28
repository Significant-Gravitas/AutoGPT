"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");

const {
  allowsPermission,
  classifyMainNavigation,
  classifyWindowOpen,
} = require("../src/navigation");

const APP = "http://127.0.0.1:17119";

test("the app's own pages open inside the app", () => {
  assert.equal(
    classifyWindowOpen({ url: `${APP}/build?flowID=1`, disposition: "foreground-tab" }, APP),
    "app",
  );
  assert.equal(classifyMainNavigation(`${APP}/library`, APP), "app");
});

test("sign-in popups stay in the app so they can report back", () => {
  // preOpenOAuthPopup() opens a blank, sized window and navigates it afterwards
  assert.equal(classifyWindowOpen({ url: "about:blank", disposition: "new-window" }, APP), "app");
  assert.equal(
    classifyWindowOpen(
      { url: "https://accounts.google.com/o/oauth2/auth", disposition: "new-window" },
      APP,
    ),
    "app",
  );
});

test("ChatGPT keeps its device-login popup and callback in the app while verification opens in the browser", () => {
  const page = fs.readFileSync(
    path.join(__dirname, "..", "..", "backend", "backend", "api", "features", "integrations", "codex_device_page.py"),
    "utf8",
  );
  assert.match(page, /<a id="verification-link" target="_blank" rel="noopener noreferrer">/);

  for (const app of [APP, "http://127.0.0.1:28019"]) {
    const login = `${app}/api/proxy/api/integrations/codex/device-login/test-login#state=test-state&user_code=test-code`;
    const callback = `${app}/auth/integrations/oauth_callback?code=test-login&state=test-state`;
    const verification = "https://auth.openai.com/codex/device";
    assert.equal(classifyWindowOpen({ url: "about:blank", disposition: "new-window" }, app), "app");
    for (const disposition of ["new-window", "foreground-tab"]) {
      assert.equal(classifyWindowOpen({ url: login, disposition }, app), "app");
      assert.equal(classifyWindowOpen({ url: callback, disposition }, app), "app");
    }
    assert.equal(classifyMainNavigation(callback, app), "app");
    assert.equal(classifyWindowOpen({ url: verification, disposition: "foreground-tab" }, app), "browser");
    assert.equal(allowsPermission({ permission: "clipboard-sanitized-write", origin: login }, app), true);
    assert.equal(allowsPermission({ permission: "clipboard-read", origin: verification }, app), false);
  }
});

test("ordinary links to other sites go to the user's browser", () => {
  assert.equal(
    classifyWindowOpen({ url: "https://agpt.co/docs", disposition: "foreground-tab" }, APP),
    "browser",
  );
  assert.equal(
    classifyWindowOpen({ url: "mailto:help@agpt.co", disposition: "foreground-tab" }, APP),
    "browser",
  );
  assert.equal(classifyMainNavigation("https://example.com/", APP), "browser");
});

test("a different port on the same host is a different site", () => {
  assert.equal(classifyMainNavigation("http://127.0.0.1:9999/", APP), "browser");
});

test("anything that is neither the app nor a web link is refused", () => {
  assert.equal(
    classifyWindowOpen({ url: "file:///etc/passwd", disposition: "foreground-tab" }, APP),
    "deny",
  );
  assert.equal(
    classifyWindowOpen({ url: "about:blank", disposition: "foreground-tab" }, APP),
    "deny",
  );
  assert.equal(classifyMainNavigation("javascript:alert(1)", APP), "deny");
  assert.equal(classifyMainNavigation("not a url", APP), "deny");
});

test("before the app has a URL nothing counts as the app", () => {
  assert.equal(classifyMainNavigation(`${APP}/`, null), "browser");
  assert.equal(allowsPermission({ permission: "notifications", origin: APP }, null), false);
});

test("the app's own pages get the microphone, notifications and the clipboard", () => {
  for (const permission of ["notifications", "clipboard-read", "clipboard-sanitized-write"]) {
    assert.equal(allowsPermission({ permission, origin: `${APP}/copilot` }, APP), true);
  }
  assert.equal(
    allowsPermission({ permission: "media", origin: APP, mediaTypes: ["audio"] }, APP),
    true,
  );
});

test("the app's own pages get nothing they do not use", () => {
  assert.equal(
    allowsPermission({ permission: "media", origin: APP, mediaTypes: ["audio", "video"] }, APP),
    false,
  );
  for (const permission of ["geolocation", "usb", "openExternal", "display-capture"]) {
    assert.equal(allowsPermission({ permission, origin: APP }, APP), false);
  }
});

test("a sign-in popup, or any other site, gets no permission at all", () => {
  for (const permission of ["media", "notifications", "clipboard-read", "geolocation"]) {
    assert.equal(
      allowsPermission({ permission, origin: "https://accounts.google.com" }, APP),
      false,
    );
  }
  assert.equal(allowsPermission({ permission: "media", origin: "not a url" }, APP), false);
});
