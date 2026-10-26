"use strict";

const assert = require("node:assert/strict");
const http = require("node:http");
const test = require("node:test");
const { once } = require("node:events");
const { createPrivacyReporter } = require("../src/telemetry");
const { DEFAULT_PREFERENCES } = require("../src/preferences");

async function collector(t, respond = (request, response) => response.end("ok")) {
  const records = [];
  const server = http.createServer((request, response) => {
    let body = "";
    request.on("data", (chunk) => { body += chunk; });
    request.on("end", () => {
      records.push({ body: JSON.parse(body), headers: request.headers, url: request.url });
      respond(request, response);
    });
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(async () => { server.closeAllConnections(); await new Promise((resolve) => server.close(resolve)); });
  return { server, records, url: `http://127.0.0.1:${server.address().port}/collect` };
}

function reporter(t, url, extra = {}) {
  const instance = createPrivacyReporter({
    preferences: { ...DEFAULT_PREFERENCES }, version: "0.1.7-readiness.20261020", platform: "win32", arch: "x64",
    destinations: { usage: url, crash: url }, allowLoopbackForTests: true, ...extra,
  });
  t.after(() => instance.dispose());
  return instance;
}

test("default-off and missing destinations send nothing", async (t) => {
  const c = await collector(t);
  const r = reporter(t, c.url);
  assert.equal(await r.reportUsage("app_started"), false);
  assert.equal(await r.reportCrash("runtime_exit"), false);
  const unconfigured = reporter(t, undefined, { preferences: { ...DEFAULT_PREFERENCES, usageStatistics: true, crashReports: true } });
  assert.equal(await unconfigured.reportUsage("app_started"), false);
  assert.deepEqual(unconfigured.configuration(), { usageStatistics: { configured: false, host: null }, crashReports: { configured: false, host: null } });
  assert.deepEqual(c.records, []);
});

test("separate consent sends only the exact allowlisted metadata to a local collector", async (t) => {
  const c = await collector(t);
  const r = reporter(t, c.url, { preferences: { ...DEFAULT_PREFERENCES, usageStatistics: true } });
  assert.equal(await r.reportUsage("app_started", { prompt: "private", token: "private", email: "private" }), true);
  assert.equal(await r.reportCrash("runtime_exit"), false);
  r.updatePreferences({ ...DEFAULT_PREFERENCES, crashReports: true });
  assert.equal(await r.reportUsage("runtime_ready"), false);
  assert.equal(await r.reportCrash("renderer_gone", new Error("private path")), true);
  assert.deepEqual(c.records.map((record) => record.body), [
    { schema: 1, appVersion: "0.1.7-readiness.20261020", platform: "win32", architecture: "x64", category: "usageStatistics", event: "app_started" },
    { schema: 1, appVersion: "0.1.7-readiness.20261020", platform: "win32", architecture: "x64", category: "crashReports", event: "renderer_gone" },
  ]);
  for (const record of c.records) {
    assert.equal(record.headers.authorization, undefined);
    assert.equal(record.headers.cookie, undefined);
    assert.equal(record.headers["user-agent"], undefined);
  }
});

test("untrusted names and metadata cannot leak arbitrary strings", async (t) => {
  const c = await collector(t);
  const r = reporter(t, c.url, { preferences: { ...DEFAULT_PREFERENCES, crashReports: true }, version: "0.1.7-private-token", platform: "/private/path", arch: "email@example.com" });
  assert.equal(await r.reportCrash(new Error("private")), false);
  assert.equal(await r.reportCrash("private"), false);
  assert.equal(await r.reportCrash("runtime_exit"), true);
  assert.deepEqual(c.records[0].body, { schema: 1, appVersion: "unknown", platform: "unknown", architecture: "unknown", category: "crashReports", event: "runtime_exit" });
});

test("unsafe endpoints are ignored and HTTP needs an explicit numeric-loopback test option", () => {
  for (const endpoint of ["http://127.0.0.1/collect", "http://example.com", "file:///tmp/report", "https://user:password@example.com", "https://example.com/?token=private", "https://example.com/#private", "not a URL"]) {
    assert.equal(createPrivacyReporter({ destinations: { usage: endpoint } }).configuration().usageStatistics.configured, false);
  }
  assert.deepEqual(createPrivacyReporter({ destinations: { usage: "https://reports.example.com/desktop" } }).configuration().usageStatistics, { configured: true, host: "reports.example.com" });
  assert.equal(createPrivacyReporter({ destinations: { usage: "http://localhost/collect" }, allowLoopbackForTests: true }).configuration().usageStatistics.configured, false);
});

test("withdrawal aborts a pending request, drops concurrent reports and never replays them", async (t) => {
  let received;
  const arrived = new Promise((resolve) => { received = resolve; });
  let disconnected;
  const closed = new Promise((resolve) => { disconnected = resolve; });
  const c = await collector(t, (request, response) => { response.on("close", disconnected); received(); });
  const r = reporter(t, c.url, { preferences: { ...DEFAULT_PREFERENCES, usageStatistics: true } });
  const first = r.reportUsage("app_started");
  await arrived;
  assert.equal(await r.reportUsage("runtime_ready"), false);
  r.updatePreferences({ ...DEFAULT_PREFERENCES });
  assert.equal(await first, false);
  await closed;
  assert.equal(await r.reportUsage("runtime_stopped"), false);
  r.updatePreferences({ ...DEFAULT_PREFERENCES, usageStatistics: true });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(c.records.length, 1);
});

test("immediate withdrawal before connection and malformed preferences remain off", async (t) => {
  const c = await collector(t);
  const r = reporter(t, c.url, { preferences: { ...DEFAULT_PREFERENCES, usageStatistics: true } });
  const first = r.reportUsage("app_started");
  r.updatePreferences({ version: 1, usageStatistics: true });
  assert.equal(await first, false);
  assert.equal(await r.reportUsage("runtime_ready"), false);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(c.records, []);
});

test("redirects and server failures are not retried", async (t) => {
  const destination = await collector(t);
  const c = await collector(t, (request, response) => { response.writeHead(302, { Location: destination.url }); response.end(); });
  const r = reporter(t, c.url, { preferences: { ...DEFAULT_PREFERENCES, usageStatistics: true } });
  assert.equal(await r.reportUsage("app_started"), false);
  assert.equal(c.records.length, 1);
  assert.equal(destination.records.length, 0);
});

test("a stalled collector times out without retaining a task or retry", { timeout: 6000 }, async (t) => {
  const c = await collector(t, () => {});
  const r = reporter(t, c.url, { preferences: { ...DEFAULT_PREFERENCES, usageStatistics: true } });
  assert.equal(await r.reportUsage("app_started"), false);
  assert.equal(c.records.length, 1);
  r.dispose();
  assert.equal(await r.reportUsage("runtime_ready"), false);
});
