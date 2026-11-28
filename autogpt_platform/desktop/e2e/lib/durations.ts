// How long the things a user waits for took on this machine: the installer,
// the first start, a later start, the upgrade and the start after it.
//
// Each is a note on the test that measured it, like every other fact these
// tests record. They are also kept together, in the order they happened, in
// `test-results/durations.md`: a table a workflow can print without reading
// the report.

import fs from "node:fs";
import path from "node:path";

import { test } from "@playwright/test";

import { STATE_DIR } from "./state";

export const DURATIONS_FILE = path.join(__dirname, "..", "test-results", "durations.md");
const RECORD = path.join(STATE_DIR, "durations.json");

interface Duration {
  what: string;
  seconds: number;
}

/** Run `work` and record how long it took under `what`. A failure is not a
 * duration: nothing is recorded, and the error is the caller's. */
export async function timed<T>(what: string, work: () => Promise<T>): Promise<T> {
  const started = Date.now();
  const result = await work();
  recordDuration(what, (Date.now() - started) / 1000);
  return result;
}

export function recordDuration(what: string, seconds: number): void {
  const rounded = Math.round(seconds);
  test.info().annotations.push({ type: `seconds: ${what}`, description: String(rounded) });
  console.log(`seconds: ${what}: ${rounded}`);
  const all = [...recorded(), { what, seconds: rounded }];
  fs.mkdirSync(STATE_DIR, { recursive: true });
  fs.writeFileSync(RECORD, JSON.stringify(all, null, 2));
  fs.mkdirSync(path.dirname(DURATIONS_FILE), { recursive: true });
  fs.writeFileSync(DURATIONS_FILE, table(all));
}

export function table(durations: Duration[]): string {
  const rows = durations.map(({ what, seconds }) => `| ${what} | ${seconds} |`);
  return ["| What | Seconds |", "| --- | ---: |", ...rows, ""].join("\n");
}

/** What this run has recorded so far. `resetState` (the first test) clears it. */
function recorded(): Duration[] {
  try {
    return JSON.parse(fs.readFileSync(RECORD, "utf8"));
  } catch {
    return [];
  }
}
