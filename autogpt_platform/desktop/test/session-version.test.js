"use strict";

const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const test = require("node:test");

for (const [version, previous, expected] of [
  ["0.0.0-dev.25", "0.0.0-dev.24", true],
  ["0.0.0-dev.24", "0.0.0-dev.25", false],
  ["1.0.0", "1.0.0-rc.1", true],
  ["1.0.0-rc.1", "1.0.0", false],
  ["1.0.0-dev.10", "1.0.0-dev.9", true],
  ["1.0.0-beta", "1.0.0-alpha", true],
  ["1.0.0-alpha.1", "1.0.0-alpha", true],
  ["1.0.0-alpha", "1.0.0-alpha.1", false],
  ["1.0.0-alpha.x", "1.0.0-alpha.9", true],
  ["1.0.0-alpha.9", "1.0.0-alpha.x", false],
  ["1.0.0+build.2", "1.0.0+build.1", false],
  ["1.0.0-rc.1+build.2", "1.0.0-rc.1+build.1", false],
  ["1.0.0-dev-name.2", "1.0.0-dev-name.1", true],
  ["2.0.0-dev.1", "1.9.9", true],
  ["1.2.3", "1.2.3", false],
]) {
  test(`upgrade ordering: ${version} > ${previous} is ${expected}`, async () => {
    const { isNewer } = await import(pathToFileURL(path.join(__dirname, "../e2e/lib/session.ts")).href);
    assert.equal(isNewer(version, previous), expected);
  });
}
