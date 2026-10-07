import { describe, expect, it } from "vitest";

import { emptyTurnLog, type LogRow, type PersistedRow } from "../turnLog";
import {
  mergeRows,
  newTurnSegment,
  turnRows,
  userSegment,
  type Segment,
} from "../turnTail";

describe("turnRows", () => {
  it("stops a stopped turn's rows at the next prompt, before that prompt has a sequence", () => {
    // A send right after a stop: the next prompt is still local when the
    // stopped turn reconciles, and its row is already persisted.
    const stopped = {
      ...newTurnSegment("turn-1", emptyTurnLog()),
      stopped: true,
    };
    const segments: Segment[] = [
      { ...userSegment(prompt("first"), "sent"), sequence: 0 },
      stopped,
      userSegment(prompt("second"), "sent"),
    ];
    const rows: PersistedRow[] = [
      { role: "user", content: "first", sequence: 0 },
      { role: "assistant", content: "Partial", sequence: 1 },
      {
        role: "assistant",
        content: "[__COPILOT_ERROR_f7a1__] Operation cancelled",
        sequence: 2,
      },
      { role: "user", content: "second", sequence: 3 },
      { role: "assistant", content: "Next reply", sequence: 4 },
    ];

    const persisted = turnRows(segments, 0, stopped, rows);

    expect(persisted?.start).toBe(1);
    expect(persisted?.rows.map((r) => r.content)).toEqual([
      "Partial",
      "[__COPILOT_ERROR_f7a1__] Operation cancelled",
    ]);
  });
});

describe("mergeRows", () => {
  it("keeps every on-screen key, and an equal row's object", () => {
    const live = [row("block-1", "Hello", 1), row("block-2", "Bye", null)];
    const persisted = [row("seq:1", "Hello", 1), row("seq:2", "Goodbye", 2)];

    const merged = mergeRows(live, persisted);

    expect(merged.map((r) => r.key)).toEqual(["block-1", "block-2"]);
    expect(merged[0]).toBe(live[0]);
    expect(merged[1]).toMatchObject({ content: "Goodbye", sequence: 2 });
  });

  it("drops streamed rows past the DB's end once an earlier row is out of place", () => {
    // The fold holds one row the DB does not, so every later index is shifted.
    const live = [
      row("block-0", "Thinking", null),
      row("block-1", "Hello", null),
      row("block-2", "Bye", null),
    ];
    const persisted = [row("seq:1", "Hello", 1), row("seq:2", "Bye", 2)];

    expect(mergeRows(live, persisted).map((r) => r.content)).toEqual([
      "Hello",
      "Bye",
    ]);
  });
});

function row(key: string, content: string, sequence: number | null): LogRow {
  return {
    key,
    role: "assistant",
    content,
    toolCalls: [],
    toolCallId: null,
    sequence,
    metadata: null,
  };
}

function prompt(text: string) {
  return {
    id: `local:${text}`,
    role: "user" as const,
    parts: [{ type: "text" as const, text }],
  };
}
