// The shell appends everything its runtime prints to <data>/logs/runtime.log
// (src/main.js), including the JSON events of the protocol in src/runtime.js.
// That file is how the tests learn the app's address without asking the app.

import fs from "node:fs";
import path from "node:path";

const POLL_MS = 1000;
const TAIL_LINES = 40;
// An app that dies before its runtime starts writes no runtime.log at all.
// Listing processes is slow on Windows, so it is not done on every poll.
const LAUNCH_GRACE_MS = 30_000;
const ALIVE_CHECK_MS = 10_000;

/** The app that was just started, as far as the wait needs to know it. */
export interface Launch {
  /** Whether any process of the app exists. */
  isRunning(): boolean;
  /** What the launcher and the shell printed. */
  output(): string;
}

interface RuntimeEvent {
  event: string;
  url?: string;
  message?: string;
  fatal?: boolean;
  graph_runs?: number;
  copilot_turns?: number;
}

/** How many lines the log has now; events before this belong to earlier starts. */
export function logPosition(dataDir: string): number {
  return readLines(dataDir).length;
}

/** The URL of the first `ready` event after `position`. Fails as soon as the
 * app is gone instead of waiting out the timeout for a log nobody writes. */
export async function waitForReady(
  dataDir: string,
  position: number,
  timeoutMs: number,
  launch?: Launch,
): Promise<string> {
  const started = Date.now();
  const deadline = started + timeoutMs;
  let nextAliveCheck = started + LAUNCH_GRACE_MS;
  const evidence = () => `${launch ? `--- launcher\n${launch.output()}\n\n` : ""}${logTails(dataDir)}`;
  for (;;) {
    // Before looking at the log, so that an app that reported and then
    // exited is judged by what it reported.
    const gone = launch !== undefined && Date.now() >= nextAliveCheck && !launch.isRunning();
    if (Date.now() >= nextAliveCheck) nextAliveCheck = Date.now() + ALIVE_CHECK_MS;
    for (const event of eventsAfter(dataDir, position)) {
      if (event.event === "ready" && typeof event.url === "string") return event.url;
      if (event.event === "error" && event.fatal) {
        throw new Error(`the runtime failed to start: ${event.message}\n${evidence()}`);
      }
    }
    if (gone) {
      const after = Math.round((Date.now() - started) / 1000);
      throw new Error(`the app exited within ${after}s without reporting ready\n${evidence()}`);
    }
    if (Date.now() > deadline) {
      throw new Error(`the runtime did not report ready within ${timeoutMs / 1000}s\n${evidence()}`);
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
  }
}

/** How many agent runs and AutoPilot turns the shell takes to be in
 * progress: what the runtime's last `activity` event after `position` said
 * (src/quit.js reads the same events), and none before the first one. */
export function inProgress(dataDir: string, position: number): number {
  const counts = eventsAfter(dataDir, position).filter((event) => event.event === "activity");
  const last = counts.at(-1);
  return last ? (last.graph_runs ?? 0) + (last.copilot_turns ?? 0) : 0;
}

/** Wait until the shell has been told that nothing is in progress. Returns
 * what it still takes to be in progress when the time is up. */
export async function waitForIdle(dataDir: string, position: number, timeoutMs: number): Promise<number> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const left = inProgress(dataDir, position);
    if (left === 0 || Date.now() > deadline) return left;
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
  }
}

/** The end of every service log, for a failure message. */
export function logTails(dataDir: string): string {
  const logs = path.join(dataDir, "logs");
  let names: string[];
  try {
    names = fs.readdirSync(logs).filter((name) => name.endsWith(".log")).sort();
  } catch {
    return `(no logs in ${logs})`;
  }
  return names
    .map((name) => {
      const lines = fs.readFileSync(path.join(logs, name), "utf8").trimEnd().split("\n");
      return `--- ${name} (last ${Math.min(lines.length, TAIL_LINES)} lines)\n${lines.slice(-TAIL_LINES).join("\n")}`;
    })
    .join("\n\n");
}

function eventsAfter(dataDir: string, position: number): RuntimeEvent[] {
  const events: RuntimeEvent[] = [];
  for (const line of readLines(dataDir).slice(position)) {
    if (!line.startsWith("{")) continue;
    try {
      const value = JSON.parse(line);
      if (value && typeof value.event === "string") events.push(value);
    } catch {
      // A log line that happens to start with a brace.
    }
  }
  return events;
}

function readLines(dataDir: string): string[] {
  try {
    const text = fs.readFileSync(path.join(dataDir, "logs", "runtime.log"), "utf8");
    // The last element is an unfinished line, or empty after a final newline.
    return text.split("\n").slice(0, -1);
  } catch {
    return [];
  }
}
