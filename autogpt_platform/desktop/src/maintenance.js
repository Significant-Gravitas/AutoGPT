"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

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
  if (!Number.isSafeInteger(pid) || pid <= 0) return true;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code !== "ESRCH";
  }
}

function operationActive(file) {
  let content;
  try {
    if (fs.lstatSync(file).isSymbolicLink()) return true;
    content = fs.readFileSync(file, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
  try {
    return processExists(JSON.parse(content).pid);
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

function claimLaunch(directory) {
  const locations = maintenancePaths(directory);
  fs.mkdirSync(path.dirname(locations.root), { recursive: true });
  const lease = path.join(
    path.dirname(locations.root),
    `.autogpt-shell-${locations.key}-${process.pid}-${crypto.randomUUID()}.json`,
  );
  const temporary = `${lease}.tmp`;
  let created = false;
  try {
    const descriptor = fs.openSync(temporary, "wx", 0o600);
    try {
      fs.writeFileSync(descriptor, JSON.stringify({ pid: process.pid }));
      fs.fsyncSync(descriptor);
    } finally {
      fs.closeSync(descriptor);
    }
    fs.renameSync(temporary, lease);
    created = true;
    requireNoRestore(locations.journal);
    if (operationActive(locations.operation)) {
      throw new Error("AutoGPT backup or restore is in progress. Wait for it to finish, then open AutoGPT again.");
    }
  } catch (error) {
    if (created) fs.rmSync(lease, { force: true });
    throw error;
  } finally {
    fs.rmSync(temporary, { force: true });
  }
  // Keep the lease through Electron's final profile writes. Maintenance ignores it only after this PID exits.
  return lease;
}

module.exports = { canonicalDirectory, claimLaunch, maintenancePaths, operationActive, processExists };
