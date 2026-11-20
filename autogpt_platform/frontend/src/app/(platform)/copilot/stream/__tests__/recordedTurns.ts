import fs from "node:fs";
import path from "node:path";

import { readSseFrames } from "../sseClient";
import type { StreamEntry } from "../turnConverter";
import type { PersistedRow } from "../turnLog";

const FIXTURE_ROOT = path.resolve(
  process.cwd(),
  "../backend/test/fixtures/copilot_stream",
);

export interface RecordedTurn {
  name: string;
  /** The frames the route writes for each stored entry, as the browser reads them. */
  sse: string[];
  /** Every row the session GET returns once the turn has ended. */
  rows: PersistedRow[];
}

/** Every turn `stream_drift/recording.py` recorded, both engines. */
export function recordedTurnNames(): string[] {
  return fs
    .readdirSync(FIXTURE_ROOT, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
}

export function loadRecordedTurn(name: string): RecordedTurn {
  const dir = path.join(FIXTURE_ROOT, name);
  const sse = fs
    .readFileSync(path.join(dir, "frames.jsonl"), "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => (JSON.parse(line) as { sse: string }).sse);
  const rows = JSON.parse(
    fs.readFileSync(path.join(dir, "rows.json"), "utf8"),
  ) as PersistedRow[];
  return { name, sse, rows };
}

/** The turn's entries, through the same SSE parser the client uses. */
export async function entriesOf(turn: RecordedTurn): Promise<StreamEntry[]> {
  const entries: StreamEntry[] = [];
  const body = new Response(turn.sse.join("") + "data: [DONE]\n\n").body!;
  await readSseFrames(body, (frame) => {
    if (frame.kind === "entry") entries.push(frame.entry);
  });
  return entries;
}
