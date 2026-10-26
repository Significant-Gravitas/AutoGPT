// What one test file leaves for the next: the account, the agent and the
// address of the first run. The files run in order in one worker, but a
// failure restarts the worker, so nothing is kept in memory.

import fs from "node:fs";
import path from "node:path";

export const STATE_DIR = path.join(__dirname, "..", ".state");
const STATE_FILE = path.join(STATE_DIR, "state.json");

export interface State {
  email: string;
  password: string;
  /** The app's address; it must not move between starts. */
  url: string;
  version: string;
  agentName: string;
  graphId: string;
  executionId: string;
}

export function resetState(): void {
  fs.rmSync(STATE_DIR, { recursive: true, force: true });
  fs.mkdirSync(STATE_DIR, { recursive: true });
}

export function saveState(values: Partial<State>): void {
  fs.mkdirSync(STATE_DIR, { recursive: true });
  fs.writeFileSync(STATE_FILE, JSON.stringify({ ...readState(), ...values }, null, 2));
}

/** Everything the first run recorded; fails when it did not get that far. */
export function loadState(): State {
  const state = readState();
  const missing = ["email", "password", "url", "version", "agentName", "graphId", "executionId"].filter(
    (name) => !(name in state),
  );
  if (missing.length > 0) {
    throw new Error(`01-first-run has to pass first (nothing recorded for: ${missing.join(", ")})`);
  }
  return state as State;
}

function readState(): Partial<State> {
  try {
    return JSON.parse(fs.readFileSync(STATE_FILE, "utf8"));
  } catch {
    return {};
  }
}
