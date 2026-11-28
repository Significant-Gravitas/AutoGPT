"use strict";

const http = require("node:http");
const https = require("node:https");
const { normalizePreferences } = require("./preferences");

const USAGE_EVENTS = Object.freeze(["app_started", "runtime_ready", "runtime_stopped", "doctor_run"]);
const CRASH_EVENTS = Object.freeze(["runtime_start_failed", "runtime_exit", "renderer_gone", "unhandled_error"]);

function destination(value, allowLoopbackForTests) {
  try {
    if (typeof value !== "string" || value.length > 2048) return null;
    const url = new URL(value);
    const loopback = allowLoopbackForTests && url.protocol === "http:" && ["127.0.0.1", "[::1]"].includes(url.hostname);
    if (url.protocol !== "https:" && !loopback) return null;
    if (url.username || url.password || url.search || url.hash) return null;
    return url;
  } catch {
    return null;
  }
}

function createPrivacyReporter({ preferences, version, platform = process.platform, arch = process.arch, destinations = {}, allowLoopbackForTests = false } = {}) {
  let consent = normalizePreferences(preferences);
  let disposed = false;
  const pending = new Map();
  const endpoints = {
    usageStatistics: destination(destinations?.usage, allowLoopbackForTests),
    crashReports: destination(destinations?.crash, allowLoopbackForTests),
  };
  const metadata = Object.freeze({
    schema: 1,
    appVersion: typeof version === "string" && /^\d{1,5}\.\d{1,5}\.\d{1,5}(?:-(?:dev|alpha|beta|rc|readiness)(?:[.-]\d{1,14})*)?$/.test(version) && version.length <= 80 ? version : "unknown",
    platform: ["win32", "darwin", "linux"].includes(platform) ? platform : "unknown",
    architecture: ["x64", "arm64", "ia32", "arm"].includes(arch) ? arch : "unknown",
  });

  function configuration() {
    return Object.fromEntries(Object.entries(endpoints).map(([key, endpoint]) => [key, {
      configured: Boolean(endpoint),
      host: endpoint?.host || null,
    }]));
  }

  function updatePreferences(next) {
    consent = normalizePreferences(next);
    for (const [category, operation] of pending) {
      if (!consent[category]) operation.abort();
    }
  }

  function dispose() {
    disposed = true;
    for (const operation of pending.values()) operation.abort();
  }

  function send(category, event, allowed) {
    if (disposed || !consent[category] || !endpoints[category] || !allowed.includes(event) || pending.has(category)) {
      return Promise.resolve(false);
    }
    const endpoint = endpoints[category];
    const body = JSON.stringify({ ...metadata, category, event });
    return new Promise((resolve) => {
      let request;
      let timer;
      let settled = false;
      function finish(sent) {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        pending.delete(category);
        resolve(sent);
      }
      function abort() {
        request?.destroy();
        finish(false);
      }
      pending.set(category, { abort });
      try {
        request = (endpoint.protocol === "https:" ? https : http).request(endpoint, {
          method: "POST",
          agent: false,
          headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) },
        }, (response) => {
          response.on("error", () => finish(false));
          response.on("end", () => finish(response.statusCode >= 200 && response.statusCode < 300));
          response.resume();
        });
        request.on("error", () => finish(false));
        timer = setTimeout(abort, 3000);
        timer.unref();
        request.end(body);
      } catch {
        abort();
      }
    });
  }

  return {
    configuration,
    updatePreferences,
    dispose,
    reportUsage: (event) => send("usageStatistics", event, USAGE_EVENTS),
    reportCrash: (event) => send("crashReports", event, CRASH_EVENTS),
  };
}

module.exports = { createPrivacyReporter, USAGE_EVENTS, CRASH_EVENTS };
