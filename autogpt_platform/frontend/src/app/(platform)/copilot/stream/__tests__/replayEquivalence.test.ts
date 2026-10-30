import { createHash } from "node:crypto";
import { beforeAll, describe, expect, it } from "vitest";

import { foldEntries, type StreamEntry } from "../turnConverter";
import {
  canonicalRows,
  emptyTurnLog,
  logRowFromPersisted,
  seedTurnLog,
  type LogRow,
} from "../turnLog";
import { diffRows } from "../turnShadow";
import {
  entriesOf,
  loadRecordedTurn,
  recordedTurnNames,
  type RecordedTurn,
} from "./recordedTurns";

const names = recordedTurnNames();

describe.each(names)("replay equivalence: %s", (name) => {
  let turn: RecordedTurn;
  let entries: StreamEntry[];
  let whole: ReturnType<typeof emptyTurnLog>;

  beforeAll(async () => {
    turn = loadRecordedTurn(name);
    entries = await entriesOf(turn);
    whole = foldEntries(emptyTurnLog(), entries);
  });

  it("the fold from 0-0 equals the rows the backend persisted", () => {
    const final = whole.checkpoints[whole.checkpoints.length - 1];
    expect(final, "the recorded turn ends on a checkpoint").toBeDefined();
    const persisted = turnRows(turn, final.sequence);

    expect(whole.protocolErrors).toEqual([]);
    expect(whole.status).toBe("finished");
    expect(final.rows).toBe(persisted.length);
    expect(diffRows(whole.rows, persisted)).toEqual([]);
    expect(whole.rows.map((r) => r.sequence)).toEqual(
      persisted.map((r) => r.sequence),
    );
  });

  it("every checkpoint's digest is the digest of the folded rows", () => {
    expect(whole.checkpoints.length).toBeGreaterThan(0);
    for (const checkpoint of whole.checkpoints) {
      expect(sha256(checkpoint.canonical)).toBe(checkpoint.digest);
    }
  });

  it("every cut point, resumed at or before its cursor, equals the whole fold", () => {
    for (let cut = 0; cut <= entries.length; cut++) {
      const head = foldEntries(emptyTurnLog(), entries.slice(0, cut));
      for (let from = 0; from <= cut; from++) {
        expect(foldEntries(head, entries.slice(from))).toEqual(whole);
      }
    }
  });

  it("the persisted rows at each checkpoint, tailed from it, equal the whole fold", () => {
    whole.checkpoints.forEach((checkpoint) => {
      const at = entries.findIndex((e) => e.entryId === checkpoint.entryId);
      const seeded = seedTurnLog({
        turnId: whole.turnId!,
        rows: persistedAt(turn, checkpoint.sequence, checkpoint.rows),
        checkpoint: {
          entry_id: checkpoint.entryId,
          rows: checkpoint.rows,
          sequence: checkpoint.sequence,
        },
      });
      const resumed = foldEntries(seeded, entries.slice(at + 1));
      expect(resumed.protocolErrors).toEqual([]);
      expect(diffRows(resumed.rows, whole.rows)).toEqual([]);
      expect(canonicalRows(resumed.rows)).toBe(canonicalRows(whole.rows));
    });
  });

  // A second connection opens at the cursor it finds, or before it; from then
  // on both deliver in their own order and race arbitrarily.
  it("two overlapping deliveries, randomly interleaved, converge", () => {
    const random = seededRandom(name.length * 7919);
    for (let trial = 0; trial < 200; trial++) {
      const applied = Math.floor(random() * (entries.length + 1));
      const opensAt = Math.floor(random() * (applied + 1));
      const merged = [
        ...entries.slice(0, applied),
        ...interleave(entries.slice(applied), entries.slice(opensAt), random),
      ];
      expect(foldEntries(emptyTurnLog(), merged)).toEqual(whole);
    }
  });
});

it("the recorded turns cover both engines", () => {
  expect(names).toEqual(
    expect.arrayContaining(["baseline-tool-turn", "sdk-late-tool-result"]),
  );
});

function turnRows(turn: RecordedTurn, fromSequence: number): LogRow[] {
  return turn.rows
    .filter((row) => (row.sequence ?? -1) >= fromSequence)
    .map(logRowFromPersisted);
}

function persistedAt(turn: RecordedTurn, sequence: number, count: number) {
  return turn.rows.filter(
    (row) =>
      (row.sequence ?? -1) >= sequence &&
      (row.sequence ?? -1) < sequence + count,
  );
}

/** Each delivery keeps its own order; the two race arbitrarily. */
function interleave<T>(a: T[], b: T[], random: () => number): T[] {
  const out: T[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length || j < b.length) {
    const takeA = j >= b.length || (i < a.length && random() < 0.5);
    out.push(takeA ? a[i++] : b[j++]);
  }
  return out;
}

function seededRandom(seed: number) {
  let state = seed % 2147483647 || 1;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function sha256(text: string) {
  return createHash("sha256").update(text, "utf8").digest("hex");
}
