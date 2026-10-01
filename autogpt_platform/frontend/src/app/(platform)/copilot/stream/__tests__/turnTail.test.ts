import { describe, expect, it } from "vitest";

import { emptyTurnLog, type PersistedRow } from "../turnLog";
import {
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

function prompt(text: string) {
  return {
    id: `local:${text}`,
    role: "user" as const,
    parts: [{ type: "text" as const, text }],
  };
}
