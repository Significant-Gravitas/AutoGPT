import type { UIDataTypes, UIMessage, UITools } from "ai";
import { describe, expect, it } from "vitest";

import { createAssistantRunJoiner, statsOfAssistantRuns } from "../renderTail";

type Msg = UIMessage<unknown, UIDataTypes, UITools>;

function message(id: string, role: Msg["role"]): Msg {
  return { id, role, parts: [{ type: "text", text: id }] };
}

describe("statsOfAssistantRuns", () => {
  it("gives a joined bubble its last turn's saved stats, as a reload does", () => {
    const messages = [
      message("u1", "user"),
      message("a1", "assistant"),
      message("a2", "assistant"),
      message("u2", "user"),
      message("a3", "assistant"),
    ];
    const stats = new Map([
      ["a1", { durationMs: 5_000, createdAt: "2026-10-24T10:00:00Z" }],
      ["a2", { durationMs: 30_000, createdAt: "2026-10-24T10:01:00Z" }],
      ["a3", { durationMs: 7_000 }],
    ]);

    const bubbles = createAssistantRunJoiner()(messages);
    const joined = statsOfAssistantRuns(messages, stats);

    expect(bubbles.map((bubble) => bubble.id)).toEqual([
      "u1",
      "a1",
      "u2",
      "a3",
    ]);
    expect(joined.get("a1")).toEqual({
      durationMs: 30_000,
      createdAt: "2026-10-24T10:01:00Z",
    });
    // A user row between them keeps two bubbles apart.
    expect(joined.get("a3")).toEqual({ durationMs: 7_000 });
  });
});
