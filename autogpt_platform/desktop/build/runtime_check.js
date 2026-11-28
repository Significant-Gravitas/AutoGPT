"use strict";

const fs = require("node:fs");
const path = require("node:path");

function inside(root, candidate) {
  const relative = path.relative(root, candidate);
  return relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

function requireFile(root, relative) {
  const file = path.resolve(root, relative);
  if (!inside(root, file) || !inside(root, fs.realpathSync(file))) {
    throw new Error(`runtime file points outside the bundle: ${relative}`);
  }
  const stat = fs.statSync(file);
  if (!stat.isFile() || stat.size === 0) throw new Error(`runtime file is empty or not a file: ${relative}`);
}

function checkRuntime(directory, platform) {
  try {
    if (fs.lstatSync(directory).isSymbolicLink()) {
      throw new Error("runtime directory must be bundled, not linked to a development directory");
    }
    const root = fs.realpathSync(directory);
    requireFile(root, "manifest.json");
    const manifest = JSON.parse(fs.readFileSync(path.join(root, "manifest.json"), "utf8"));
    const entry = manifest[platform] || manifest.default;
    if (!entry || typeof entry.command !== "string" || !entry.command ||
        path.posix.isAbsolute(entry.command) || path.win32.isAbsolute(entry.command)) {
      throw new Error(`runtime manifest needs a bundled command for ${platform}`);
    }
    const exe = platform === "win32" ? ".exe" : "";
    const script = platform === "win32" ? ".bat" : "";
    for (const file of [
      entry.command,
      "autogpt_desktop/__main__.py",
      "backend/backend/app.py",
      "backend/schema.prisma",
      "frontend/server.js",
      `postgres/bin/postgres${exe}`,
      `valkey/valkey-server${exe}`,
      `erlang/bin/erl${exe}`,
      `rabbitmq/sbin/rabbitmq-server${script}`,
      "prisma/node_modules/prisma/build/index.js",
      "assets/00-init.sql",
    ]) requireFile(root, file);
  } catch (error) {
    throw new Error(`AutoGPT runtime is incomplete at ${directory}. Rebuild the runtime before packaging. ${error.message}`, { cause: error });
  }
}

function beforePack(context) {
  checkRuntime(path.join(context.packager.info.appDir, "build", "runtime"), context.electronPlatformName);
}

function afterPack(context) {
  const resources = context.electronPlatformName === "darwin"
    ? path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`, "Contents", "Resources")
    : path.join(context.appOutDir, "resources");
  checkRuntime(path.join(resources, "runtime"), context.electronPlatformName);
}

module.exports = { checkRuntime, beforePack, afterPack };
