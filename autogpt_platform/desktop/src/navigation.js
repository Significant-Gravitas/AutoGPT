"use strict";

// Where a new window or a navigation should go.
//
//   "app"      stays inside the desktop app
//   "browser"  is handed to the user's default browser
//   "deny"     goes nowhere
//
// Two things must stay inside the app. Its own pages, obviously. And popups:
// connecting an integration opens the provider's sign-in page in a popup that
// reports back over BroadcastChannel / postMessage / localStorage
// (frontend/src/lib/oauth-popup.ts), which only works when the popup shares
// the app's browser profile. Ordinary links to other sites (docs, the
// marketplace, "contact us") are better off in the user's real browser.

const EXTERNAL_SCHEMES = new Set(["http:", "https:", "mailto:"]);

function sameOrigin(url, appUrl) {
  try {
    return Boolean(appUrl) && new URL(url).origin === new URL(appUrl).origin;
  } catch {
    return false;
  }
}

function isExternalLink(url) {
  try {
    return EXTERNAL_SCHEMES.has(new URL(url).protocol);
  } catch {
    return false;
  }
}

// `disposition` is Electron's: "new-window" for window.open() with popup
// features (a size, or popup=true), "foreground-tab" for target="_blank".
function classifyWindowOpen({ url, disposition }, appUrl) {
  if (sameOrigin(url, appUrl)) return "app";
  if (disposition === "new-window") return "app";
  return isExternalLink(url) ? "browser" : "deny";
}

// Navigation of the main window itself: it never leaves the app.
function classifyMainNavigation(url, appUrl) {
  if (sameOrigin(url, appUrl)) return "app";
  return isExternalLink(url) ? "browser" : "deny";
}

// Electron grants every permission a page asks for unless told otherwise, and
// a sign-in popup can be any site at all. Only the app's own pages get
// anything, and only what they use: the microphone (voice input, not the
// camera), notifications, the clipboard and fullscreen.
const APP_PERMISSIONS = new Set([
  "media",
  "notifications",
  "clipboard-read",
  "clipboard-sanitized-write",
  "fullscreen",
]);

function allowsPermission({ permission, origin, mediaTypes = [] }, appUrl) {
  if (!sameOrigin(origin, appUrl)) return false;
  if (permission === "media" && mediaTypes.includes("video")) return false;
  return APP_PERMISSIONS.has(permission);
}

module.exports = { classifyWindowOpen, classifyMainNavigation, allowsPermission };
