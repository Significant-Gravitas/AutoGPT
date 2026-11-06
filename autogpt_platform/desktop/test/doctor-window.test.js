"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const { isDoctorSender, doctorTab } = require("../src/doctor-window");

test("Doctor IPC accepts only its own packaged top-level frame", () => {
  const url = "file:///C:/AutoGPT/src/doctor.html";
  const frame = { url };
  const contents = { mainFrame: frame, getURL: () => url, isDestroyed: () => false };
  const event = { sender: contents, senderFrame: frame };
  assert.equal(isDoctorSender(event, contents, url), true);
  assert.equal(isDoctorSender({ ...event, sender: {} }, contents, url), false);
  assert.equal(isDoctorSender({ ...event, senderFrame: { url } }, contents, url), false);
  assert.equal(isDoctorSender(event, contents, "https://example.com"), false);
  frame.url = "https://example.com";
  assert.equal(isDoctorSender(event, contents, url), false);
});

test("Doctor tab selection is a fixed local allowlist", () => {
  for (const tab of ["doctor", "features", "privacy", "desktop"]) assert.equal(doctorTab(tab), tab);
  for (const tab of [null, {}, "../../config/settings.env", "javascript:alert(1)"]) {
    assert.equal(doctorTab(tab), "doctor");
  }
});
