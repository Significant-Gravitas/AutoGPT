"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../src/main.js"), "utf8");
const method = source.slice(source.indexOf("async function doctorConnectionStatus()"), source.indexOf("function showDoctor("));

async function check(response) {
  const origin = "http://127.0.0.1:24754";
  return vm.runInNewContext(`${method}\ndoctorConnectionStatus()`, {
    appUrl: origin, runtimeExited: () => false, URL, setTimeout, clearTimeout,
    mainWindow: { webContents: {
      getURL: () => `${origin}/copilot`,
      executeJavaScript: code => vm.runInNewContext(code, { AbortSignal, fetch: async () => response }),
    } },
  });
}

test("Doctor reports a connection only for ready and selectable offers", async () => {
  const ready = { provider_family: "openai", state: "ready", selectable: true, lock_reason: null };
  for (const provider_family of ["openai", "autogpt", "microsoft"]) {
    assert.equal(await check({ ok: true, json: async () => ({ offers: [{ ...ready, provider_family }] }) }), "connected");
  }
  for (const offer of [null, {}, { provider_family: "openai" }, { ...ready, selectable: false },
    { ...ready, state: "unavailable" }, { ...ready, lock_reason: "needs_reconnect" }]) {
    assert.equal(await check({ ok: true, json: async () => ({ offers: [offer] }) }), "not-connected");
  }
});

test("Doctor distinguishes owner sign-in, inaccessible service and malformed offers", async () => {
  for (const status of [401, 403]) assert.equal(await check({ status }), "sign-in-required");
  assert.equal(await check({ status: 503, ok: false }), "unavailable");
  assert.equal(await check({ ok: true, json: async () => ({}) }), "unknown");
});
