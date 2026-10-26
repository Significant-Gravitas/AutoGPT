// Which processes are running, by executable path: the only evidence that a
// quit, an upgrade or an uninstall left nothing behind.

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

export interface RunningProcess {
  pid: number;
  /** Full path of the executable; empty when the OS will not say. */
  executable: string;
  command: string;
}

export function listProcesses(): RunningProcess[] {
  if (process.platform === "win32") return listWindows();
  if (process.platform === "darwin") return listMac();
  return listLinux();
}

export function isUnder(file: string, root: string): boolean {
  const insensitive = process.platform !== "linux";
  const normalize = (value: string) => {
    const resolved = path.resolve(value);
    return insensitive ? resolved.toLowerCase() : resolved;
  };
  return normalize(file).startsWith(normalize(root) + path.sep);
}

export function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    // EPERM: it exists and belongs to someone else.
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

/** The services the runtime recorded as started (runtime/autogpt_desktop/process.py). */
export function recordedPids(dataDir: string): number[] {
  try {
    const entries = JSON.parse(fs.readFileSync(path.join(dataDir, "run", "children.json"), "utf8"));
    return Array.isArray(entries) ? entries.map((entry) => Number(entry.pid)).filter(Number.isInteger) : [];
  } catch {
    return [];
  }
}

export function describe(processes: RunningProcess[]): string {
  return processes.map((found) => `${found.pid} ${found.executable || found.command}`).join("\n");
}

function listWindows(): RunningProcess[] {
  const output = execFileSync(
    "powershell.exe",
    [
      "-NoProfile",
      "-Command",
      "Get-CimInstance Win32_Process | Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress",
    ],
    { encoding: "utf8", maxBuffer: 256 * 1024 * 1024 },
  );
  const rows = JSON.parse(output) as Array<{
    ProcessId: number;
    ExecutablePath: string | null;
    CommandLine: string | null;
  }>;
  return rows.map((row) => ({
    pid: row.ProcessId,
    executable: row.ExecutablePath || "",
    command: row.CommandLine || "",
  }));
}

function listMac(): RunningProcess[] {
  // `comm` is the executable's full path on macOS; `args` starts with it too,
  // but cannot be split reliably when the path has spaces.
  const executables = psColumn("comm=");
  const commands = psColumn("args=");
  return [...executables].map(([pid, executable]) => ({
    pid,
    executable,
    command: commands.get(pid) || executable,
  }));
}

function psColumn(column: string): Map<number, string> {
  const output = execFileSync("ps", ["-axww", "-o", `pid=,${column}`], {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  const values = new Map<number, string>();
  for (const line of output.split("\n")) {
    const match = /^\s*(\d+)\s+(.*)$/.exec(line);
    if (match) values.set(Number(match[1]), match[2]);
  }
  return values;
}

function listLinux(): RunningProcess[] {
  const found: RunningProcess[] = [];
  for (const name of fs.readdirSync("/proc")) {
    if (!/^\d+$/.test(name)) continue;
    try {
      // A replaced or removed executable reads as "<path> (deleted)".
      const executable = fs.readlinkSync(`/proc/${name}/exe`).replace(/ \(deleted\)$/, "");
      const command = fs.readFileSync(`/proc/${name}/cmdline`, "utf8").replace(/\0/g, " ").trim();
      found.push({ pid: Number(name), executable, command });
    } catch {
      // Gone already, a kernel thread, or another user's process.
    }
  }
  return found;
}
