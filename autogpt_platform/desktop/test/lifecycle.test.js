"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");

const { hearsRuntime } = require("../src/lifecycle");

const old = { name: "old" };
const next = { name: "next" };
const ready = { event: "ready", url: "http://127.0.0.1:18473" };
const failure = { event: "error", fatal: true, message: "the executor keeps stopping" };
const finishing = { event: "progress", step: "stopping", message: "Finishing a database update", grace_seconds: 960 };

test("the running runtime is heard", () => {
  for (const event of [ready, failure, finishing]) {
    assert.equal(hearsRuntime({ current: old, sender: old, restarting: false, event }), true);
  }
});

test("a runtime being stopped for a restart cannot open the window or fail the new start", () => {
  assert.equal(hearsRuntime({ current: old, sender: old, restarting: true, event: ready }), false);
  assert.equal(hearsRuntime({ current: old, sender: old, restarting: true, event: failure }), false);
  const starting = { event: "progress", step: "services", message: "Starting AutoGPT" };
  assert.equal(hearsRuntime({ current: old, sender: old, restarting: true, event: starting }), false);
});

test("its request for more time to stop is still shown", () => {
  assert.equal(hearsRuntime({ current: old, sender: old, restarting: true, event: finishing }), true);
});

test("once it is replaced, nothing it says is heard", () => {
  for (const event of [ready, failure, finishing]) {
    assert.equal(hearsRuntime({ current: next, sender: old, restarting: false, event }), false);
  }
  assert.equal(hearsRuntime({ current: next, sender: next, restarting: false, event: ready }), true);
});
