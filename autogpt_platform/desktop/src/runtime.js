"use strict";

// The shell knows nothing about Postgres, Python or ports. It starts one
// runtime process and listens to it. The runtime reports its state as one JSON
// object per stdout line:
//
//   {"event": "progress", "step": "postgres", "message": "Starting the database"}
//   {"event": "ready", "url": "http://127.0.0.1:43117"}
//   {"event": "error", "message": "...", "fatal": true}
//   {"event": "timing", "phase": "queue", "seconds": 6.3}
//   {"event": "activity", "graph_runs": 2, "copilot_turns": 0}
//
// `timing` says how long a part of the start took, and once more for the
// whole of it ("phase": "total", with every phase in "phases"). It is for
// the log and for the smoke test; the window shows nothing for it.
// `activity` says how many agent runs and AutoPilot turns are in progress,
// whenever that changes; quit.js is what it is for.
//
// Anything on stdout that is not a JSON object is treated as a log line.
// The shell asks the runtime to stop by closing its stdin. Signals do not
// reach a console-less child on Windows, and a closed pipe also covers the
// shell crashing: the runtime sees EOF either way and shuts its services down.
// What was running then resumes on the next start.
//
// When the user chose to stop what is running, the shell says so first, on
// a line of its own, and leaves stdin open:
//
//   {"command": "stop", "drain": true}
//
// The runtime then asks the runs to end, waits a while for them, and stops.
// Closing stdin meanwhile is the second request: it waits for nothing more.
//
// A runtime that has not exited some time after being asked to is killed. One
// that knows its stop will be slow says how long it needs, on any event:
//
//   {"event": "progress", "message": "Finishing a database update", "grace_seconds": 960}

const { spawn } = require("node:child_process");
const { EventEmitter } = require("node:events");
const fs = require("node:fs");
const path = require("node:path");
const readline = require("node:readline");

// A healthy runtime stops in a few seconds, and at worst takes about 45: the
// broker is given 30 to exit before it is killed. Past this it is hung.
const STOP_GRACE_MS = 60_000;
const MAX_GRACE_SECONDS = 30 * 60;
const STOP_THE_RUNS = { command: "stop", drain: true };

class Runtime extends EventEmitter {
  constructor({
    command,
    args = [],
    env = {},
    cwd,
    logFile,
    registryFile,
    stopGraceMs = STOP_GRACE_MS,
  }) {
    super();
    this.command = command;
    this.args = args;
    this.env = env;
    this.cwd = cwd;
    this.logFile = logFile;
    this.registryFile = registryFile;
    this.stopGraceMs = stopGraceMs;
    this.child = null;
    this.stopping = false;
    this.exited = null;
    this.hasExited = false;
    this.killTimer = null;
  }

  start() {
    const log = this.logFile
      ? fs.createWriteStream(this.logFile, { flags: "a" })
      : null;
    this.log = log;
    let stdioClosed = false;
    let settled = false;
    let logFailed = false;
    log?.on("error", () => {
      if (logFailed) return;
      logFailed = true;
      this.emit("log", "Runtime file logging is unavailable.");
    });
    const writeLog = (line) => {
      if (log && !logFailed && !log.destroyed && !log.writableEnded) log.write(line);
    };
    const finishLog = () => {
      if (stdioClosed && settled && log && !log.destroyed && !log.writableEnded) log.end();
    };
    this.child = spawn(this.command, this.args, {
      cwd: this.cwd,
      env: { ...process.env, ...this.env },
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    this.exited = new Promise((resolve) => {
      const settle = async (code, signal) => {
        if (this.hasExited) return;
        this.hasExited = true;
        clearTimeout(this.killTimer);
        // A runtime that stopped its services also removed the list of them,
        // so this only finds work after a crash or a kill.
        if (this.child.pid !== undefined) {
          try {
            await killRecorded(this.registryFile, {
              command: this.command, cwd: this.cwd, env: { ...process.env, ...this.env },
            });
          } catch (error) {
            const message = `Runtime cleanup could not complete: ${error.message}`;
            writeLog(`${message}\n`);
            this.emit("log", message);
          }
        }
        settled = true;
        finishLog();
        this.emit("exit", { code, signal, expected: this.stopping });
        resolve({ code, signal });
      };
      this.child.once("exit", settle);
      this.child.once("close", () => {
        stdioClosed = true;
        finishLog();
      });
      // A runtime that died took the pipe with it; a request written to it
      // then is lost, not an error of the shell's.
      this.child.stdin?.on("error", () => {});
      this.child.once("error", (error) => {
        this.emit("event", { event: "error", message: error.message, fatal: true });
        // A process that could not be started never exits either.
        if (this.child.pid === undefined) settle(null, null);
      });
    });

    readline.createInterface({ input: this.child.stdout }).on("line", (line) => {
      writeLog(`${line}\n`);
      const event = parseEvent(line);
      if (!event) return this.emit("log", line);
      if (this.stopping && Number.isFinite(event.grace_seconds)) {
        this.killAfter(Math.min(event.grace_seconds, MAX_GRACE_SECONDS) * 1000);
      }
      this.emit("event", event);
    });
    readline.createInterface({ input: this.child.stderr }).on("line", (line) => {
      writeLog(`${line}\n`);
      this.emit("log", line);
    });
  }

  // `stopRuns`: end what is in progress first (see the top of this file).
  // The runtime then says how long that may take (`grace_seconds`). Called
  // again while the runtime is stopping, whichever way, it tells it to wait
  // for nothing more.
  async stop({ stopRuns = false } = {}) {
    if (!this.child) return;
    if (this.hasExited) {
      await this.exited;
      return;
    }
    const again = this.stopping;
    // Before anything is written: a request for more time is honoured only
    // from a runtime that is being stopped.
    this.stopping = true;
    const stdin = this.child.stdin;
    if (stopRuns && !again && stdin?.writable) stdin.write(`${JSON.stringify(STOP_THE_RUNS)}\n`);
    else stdin?.end();
    if (!again) this.killAfter(this.stopGraceMs);
    await this.exited;
  }

  killAfter(milliseconds) {
    clearTimeout(this.killTimer);
    this.killTimer = setTimeout(() => this.child.kill("SIGKILL"), milliseconds);
  }
}

// A runtime that crashed or had to be killed cannot stop its services. It
// keeps a list of them on disk for exactly this case (Windows also has a Job
// Object that takes them down with it; other systems rely on this).
//
// Outside Windows each service leads its own process group, and the group is
// what gets killed: RabbitMQ's entry is a shell script, and killing only the
// script would leave the broker it started running.
async function killRecorded(registryFile, { command, cwd, env } = {}) {
  if (!registryFile || !fs.existsSync(registryFile)) return;
  if (!command) throw new Error("The runtime helper is required for verified cleanup.");
  await new Promise((resolve, reject) => {
    const helper = spawn(command, [
      "-B", "-m", "autogpt_desktop.reap", "--data-dir", path.dirname(path.dirname(registryFile)),
    ], {
      cwd, env, stdio: "ignore", windowsHide: true, timeout: 15_000, killSignal: "SIGKILL",
    });
    helper.once("error", reject);
    helper.once("close", (code, signal) => {
      if (code === 0 && signal === null) resolve();
      else reject(new Error("Verified runtime cleanup failed; its registry is retained for the next start."));
    });
  });
}

function parseEvent(line) {
  if (!line.startsWith("{")) return null;
  try {
    const value = JSON.parse(line);
    return value && typeof value.event === "string" ? value : null;
  } catch {
    return null;
  }
}

module.exports = { Runtime, parseEvent, killRecorded };
