"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { test } = require("node:test");
const { checkRuntime } = require("../build/runtime_check");
const config = require("../electron-builder.config");

function fixture(t, platform) {
  const project = fs.mkdtempSync(path.join(os.tmpdir(), "autogpt-package-check-"));
  t.after(() => fs.rmSync(project, { recursive: true, force: true }));
  const runtime = path.join(project, "build", "runtime");
  const files = [
    "autogpt_desktop/__main__.py", "backend/backend/app.py", "backend/schema.prisma",
    "frontend/server.js", "prisma/node_modules/prisma/build/index.js", "assets/00-init.sql",
    ...(platform === "win32" ? [
      "python/python.exe", "postgres/bin/postgres.exe", "valkey/valkey-server.exe",
      "erlang/bin/erl.exe", "rabbitmq/sbin/rabbitmq-server.bat",
    ] : [
      "python/bin/python3", "postgres/bin/postgres", "valkey/valkey-server",
      "erlang/bin/erl", "rabbitmq/sbin/rabbitmq-server",
    ]),
  ];
  for (const file of files) {
    fs.mkdirSync(path.dirname(path.join(runtime, file)), { recursive: true });
    fs.writeFileSync(path.join(runtime, file), "fixture\n");
  }
  fs.writeFileSync(path.join(runtime, "manifest.json"), JSON.stringify({
    win32: { command: "python/python.exe" }, default: { command: "python/bin/python3" },
  }));
  const context = {
    electronPlatformName: platform,
    appOutDir: path.join(project, "dist", "unpacked"),
    packager: { info: { appDir: project }, appInfo: { productFilename: "AutoGPT-test" } },
  };
  const packed = platform === "darwin"
    ? path.join(context.appOutDir, "AutoGPT-test.app", "Contents", "Resources", "runtime")
    : path.join(context.appOutDir, "resources", "runtime");
  return { runtime, context, packed, files };
}

for (const platform of ["win32", "darwin", "linux"]) {
  test(`${platform}: packaging rejects missing source runtime and incomplete copied app`, (t) => {
    const { runtime, context, packed } = fixture(t, platform);
    config.beforePack(context);
    assert.throws(() => config.afterPack(context), /runtime is incomplete/);
    fs.cpSync(runtime, packed, { recursive: true });
    config.afterPack(context);
    fs.unlinkSync(path.join(packed, "manifest.json"));
    assert.throws(() => config.afterPack(context), /manifest.json/);
    fs.unlinkSync(path.join(runtime, "manifest.json"));
    assert.throws(() => config.beforePack(context), /manifest.json/);
  });
}

test("a manifest alone cannot make a partial runtime pass packaging", (t) => {
  const { runtime, files } = fixture(t, "win32");
  for (const file of files) {
    const target = path.join(runtime, file);
    const bytes = fs.readFileSync(target);
    fs.unlinkSync(target);
    assert.throws(() => checkRuntime(runtime, "win32"), /runtime is incomplete/, file);
    fs.writeFileSync(target, bytes);
  }
  fs.writeFileSync(path.join(runtime, "frontend/server.js"), "");
  assert.throws(() => checkRuntime(runtime, "win32"), /empty or not a file/);
});

test("an installed runtime must not depend on a developer's Python outside the bundle", (t) => {
  const { runtime } = fixture(t, "win32");
  for (const command of ["../python.exe", "C:\\Python\\python.exe", "/usr/bin/python3", ""]) {
    fs.writeFileSync(path.join(runtime, "manifest.json"), JSON.stringify({ win32: { command } }));
    assert.throws(() => checkRuntime(runtime, "win32"), /runtime is incomplete/, command);
  }
});

test("a packaged app cannot refer to the developer runtime through a directory link", (t) => {
  const { runtime, context, packed } = fixture(t, "win32");
  fs.mkdirSync(path.dirname(packed), { recursive: true });
  fs.symlinkSync(runtime, packed, process.platform === "win32" ? "junction" : "dir");
  assert.throws(() => config.afterPack(context), /not linked to a development directory/);
});
