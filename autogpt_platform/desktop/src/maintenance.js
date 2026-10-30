"use strict";

const crypto = require("node:crypto");
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { readRuntimeManifest } = require("./paths");

function resolvedDirectory(directory) {
  let current = path.resolve(directory);
  const missing = [];
  let links = 0;
  for (;;) {
    try {
      current = path.join(fs.realpathSync.native(current), ...missing);
      break;
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
      try {
        if (fs.lstatSync(current).isSymbolicLink()) {
          if (++links > 40) throw new Error("The data directory contains too many symbolic links.");
          current = path.resolve(path.dirname(current), fs.readlinkSync(current));
          continue;
        }
      } catch (linkError) {
        if (linkError.code !== "ENOENT") throw linkError;
      }
      const parent = path.dirname(current);
      if (parent === current) throw error;
      missing.unshift(path.basename(current));
      current = parent;
    }
  }
  return current;
}

function canonicalDirectory(directory, platform = process.platform) {
  const resolved = resolvedDirectory(directory);
  if (platform === "win32") return resolved.toLowerCase();
  if (platform === "darwin") return resolved.normalize("NFC").toLowerCase();
  return resolved;
}

function maintenancePaths(directory) {
  const root = resolvedDirectory(directory);
  const key = crypto.createHash("sha256").update(canonicalDirectory(root), "utf8").digest("hex").slice(0, 24);
  return {
    root,
    key,
    operation: path.join(path.dirname(root), `.autogpt-operation-${key}.json`),
    journal: path.join(path.dirname(root), `.autogpt-restore-${key}.json`),
  };
}

function processExists(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0 || pid > 2 ** 31 - 1) return true;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code !== "ESRCH";
  }
}

function processStartProbe({
  runtime,
  platform = process.platform,
  run = spawnSync,
} = {}) {
  const deadline = performance.now() + 15_000;
  return (pid) => {
    const timeout = Math.floor(deadline - performance.now());
    if (timeout <= 0) throw new Error("Process identity inspection timed out.");
    const { command } = readRuntimeManifest(runtime, platform);
    const code = "import sys,json; sys.path.insert(0,sys.argv[1]); " +
      "from autogpt_desktop.maintenance import process_started; " +
      "print(json.dumps(process_started(int(sys.argv[2]))))";
    const env = Object.fromEntries(Object.entries(process.env).filter(([name]) =>
      ["systemroot", "windir"].includes(name.toLowerCase()),
    ));
    const result = run(command, ["-I", "-B", "-c", code, path.resolve(runtime), String(pid)], {
      cwd: path.resolve(runtime), env, windowsHide: true, shell: false, timeout,
      maxBuffer: 4096, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"],
    });
    if (result.error || result.status !== 0 || result.signal) {
      throw new Error("Process identity could not be inspected.");
    }
    const started = JSON.parse(result.stdout);
    if (started !== null && (!Number.isFinite(started) || started <= 0)) {
      throw new Error("Invalid process identity.");
    }
    return started;
  };
}

function operationActive(file, processStarted) {
  let content;
  try {
    if (fs.lstatSync(file).isSymbolicLink()) return true;
    content = fs.readFileSync(file, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
  try {
    const owner = JSON.parse(content);
    if (!Number.isSafeInteger(owner.pid) || owner.pid <= 0 || owner.pid > 2 ** 31 - 1) return true;
    if (!processExists(owner.pid)) return false;
    if (!Number.isFinite(owner.started) || owner.started <= 0) return true;
    const started = processStarted(owner.pid);
    if (started === null) return false;
    if (!Number.isFinite(started) || started <= 0) return true;
    return started === owner.started;
  } catch {
    return true;
  }
}

function requireNoRestore(journal) {
  try {
    fs.lstatSync(journal);
  } catch (error) {
    if (error.code === "ENOENT") return;
    throw error;
  }
  throw new Error(
    `An interrupted restore needs recovery before AutoGPT can open this data folder. ` +
      `Keep all data folders and inspect the recovery record at ${journal}.`,
  );
}

function claimLaunch(directory, { runtime, processStarted = processStartProbe({ runtime }) } = {}) {
  const started = processStarted(process.pid);
  if (!Number.isFinite(started) || started <= 0) {
    throw new Error("AutoGPT could not verify its process identity.");
  }
  const locations = maintenancePaths(directory);
  fs.mkdirSync(path.dirname(locations.root), { recursive: true });
  removeDeadLeases(locations, processStarted);
  const lease = path.join(
    path.dirname(locations.root),
    `.autogpt-shell-${locations.key}-${process.pid}-${crypto.randomUUID()}.json`,
  );
  const temporary = `${lease}.tmp`;
  let created = false;
  try {
    const descriptor = fs.openSync(temporary, "wx", 0o600);
    try {
      fs.writeFileSync(descriptor, JSON.stringify({ pid: process.pid, started }));
      fs.fsyncSync(descriptor);
    } finally {
      fs.closeSync(descriptor);
    }
    fs.renameSync(temporary, lease);
    created = true;
    requireNoRestore(locations.journal);
    if (operationActive(locations.operation, processStarted)) {
      throw new Error("AutoGPT backup or restore is in progress. Wait for it to finish, then open AutoGPT again.");
    }
  } catch (error) {
    if (created) fs.rmSync(lease, { force: true });
    throw error;
  } finally {
    fs.rmSync(temporary, { force: true });
  }
  // Keep exclusion through Electron's final writes; later checks compare PID and native birth time.
  return lease;
}

function removeDeadLeases(locations, processStarted) {
  const parent = path.dirname(locations.root);
  const prefix = `.autogpt-shell-${locations.key}-`;
  let names;
  try { names = fs.readdirSync(parent); }
  catch { return; }
  for (const name of names) {
    if (!name.startsWith(prefix) || !name.endsWith(".json")) continue;
    const file = path.join(parent, name);
    try {
      const info = fs.lstatSync(file);
      if (!info.isFile() || info.isSymbolicLink() || info.nlink !== 1) continue;
      if (!operationActive(file, processStarted)) fs.unlinkSync(file);
    } catch {
      // Unreadable or uncertain leases remain for maintenance to reject.
    }
  }
}

module.exports = { canonicalDirectory, claimLaunch, maintenancePaths, operationActive, processExists, processStartProbe };
