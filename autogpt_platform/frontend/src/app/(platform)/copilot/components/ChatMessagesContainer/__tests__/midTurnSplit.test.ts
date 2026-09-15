import type { UIDataTypes, UIMessage, UITools } from "ai";
import { describe, expect, it } from "vitest";

import {
  isMidTurnSegmentRow,
  messagesCarryDrainedText,
  splitMessagesAtDrainHints,
} from "../midTurnSplit";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;
type Part = ChatMessage["parts"][number];

function toolPart(name: string): Part {
  return {
    type: `tool-${name}`,
    toolCallId: `call-${name}`,
    state: "output-available",
    input: {},
    output: "ok",
  } as unknown as Part;
}

function textPart(text: string): Part {
  return { type: "text", text, state: "done" } as Part;
}

function hintPart(messages?: { id: string; content: string }[]): Part {
  return {
    type: "data-pending-drained",
    id: "hint",
    data:
      messages === undefined
        ? { drainedCount: 1 }
        : { drainedCount: messages.length, messages },
  } as unknown as Part;
}

function assistant(id: string, parts: Part[]): ChatMessage {
  return { id, role: "assistant", parts };
}

function user(id: string, text: string): ChatMessage {
  return { id, role: "user", parts: [textPart(text)] };
}

const PROMPT = user("user-1", "plan my week");

describe("splitMessagesAtDrainHints", () => {
  it("returns the same array when no message carries a hint", () => {
    const messages = [
      PROMPT,
      assistant("a1", [toolPart("read"), textPart("done")]),
    ];

    expect(splitMessagesAtDrainHints(messages)).toBe(messages);
    expect(messagesCarryDrainedText(messages)).toBe(false);
  });

  it("does not split on a hint from a backend that sends no text", () => {
    const messages = [
      PROMPT,
      assistant("a1", [toolPart("read"), hintPart(), toolPart("write")]),
    ];

    expect(splitMessagesAtDrainHints(messages)).toBe(messages);
    expect(messagesCarryDrainedText(messages)).toBe(false);
  });

  it("splits one turn into chain, follow-up bubble and the work after it", () => {
    const rows = splitMessagesAtDrainHints([
      PROMPT,
      assistant("a1", [
        toolPart("read"),
        hintPart([{ id: "pm-1", content: "also check Friday" }]),
        toolPart("write"),
        textPart("done"),
      ]),
    ]);

    expect(rows.map((row) => [row.id, row.role])).toEqual([
      ["user-1", "user"],
      ["a1#seg0", "assistant"],
      ["midturn-pm-1", "user"],
      ["a1", "assistant"],
    ]);
    // The hint itself never reaches a segment — it is the delimiter.
    expect(rows[1].parts).toEqual([toolPart("read")]);
    expect(rows[2].parts).toEqual([
      { type: "text", text: "also check Friday", state: "done" },
    ]);
    expect(rows[3].parts).toEqual([toolPart("write"), textPart("done")]);
  });

  it("keeps the original id on the last segment only", () => {
    const rows = splitMessagesAtDrainHints([
      PROMPT,
      assistant("a1", [
        toolPart("read"),
        hintPart([{ id: "pm-1", content: "one" }]),
        toolPart("write"),
      ]),
    ]);

    expect(rows.filter((row) => row.id === "a1")).toHaveLength(1);
    expect(isMidTurnSegmentRow(rows[1])).toBe(true);
    expect(isMidTurnSegmentRow(rows[3])).toBe(false);
  });

  it("renders one bubble per message when a batch drains at once", () => {
    const rows = splitMessagesAtDrainHints([
      PROMPT,
      assistant("a1", [
        toolPart("read"),
        hintPart([
          { id: "pm-1", content: "first" },
          { id: "pm-2", content: "second" },
        ]),
        toolPart("write"),
      ]),
    ]);

    expect(rows.map((row) => row.id)).toEqual([
      "user-1",
      "a1#seg0",
      "midturn-pm-1",
      "midturn-pm-2",
      "a1",
    ]);
  });

  it("keeps both follow-ups when two hints land back to back", () => {
    const rows = splitMessagesAtDrainHints([
      PROMPT,
      assistant("a1", [
        toolPart("read"),
        hintPart([{ id: "pm-1", content: "first" }]),
        hintPart([{ id: "pm-2", content: "second" }]),
        toolPart("write"),
      ]),
    ]);

    // The second hint opens on an empty segment: it must still draw its
    // bubble, and must not cut an empty assistant row between the two.
    expect(rows.map((row) => row.id)).toEqual([
      "user-1",
      "a1#seg0",
      "midturn-pm-1",
      "midturn-pm-2",
      "a1",
    ]);
    expect(rows[1].parts).toEqual([toolPart("read")]);
    expect(rows[4].parts).toEqual([toolPart("write")]);
  });

  it("gives consecutive id-less hints distinct fallback row ids", () => {
    const rows = splitMessagesAtDrainHints([
      PROMPT,
      assistant("a1", [
        toolPart("read"),
        hintPart([{ id: "", content: "first" }]),
        hintPart([{ id: "", content: "second" }]),
      ]),
    ]);

    const followUps = rows.filter(
      (row) => row.role === "user" && row !== PROMPT,
    );
    expect(new Set(followUps.map((row) => row.id)).size).toBe(2);
  });

  it("splits again for every later drain in the same turn", () => {
    const rows = splitMessagesAtDrainHints([
      PROMPT,
      assistant("a1", [
        toolPart("read"),
        hintPart([{ id: "pm-1", content: "first" }]),
        toolPart("write"),
        hintPart([{ id: "pm-2", content: "second" }]),
        textPart("done"),
      ]),
    ]);

    expect(rows.map((row) => row.id)).toEqual([
      "user-1",
      "a1#seg0",
      "midturn-pm-1",
      "a1#seg1",
      "midturn-pm-2",
      "a1",
    ]);
  });

  it("ignores a hint that lands before the turn has drawn anything", () => {
    // The turn-start drain folds the queued text into the opening prompt, so
    // splitting there would show it twice.
    const messages = [
      PROMPT,
      assistant("a1", [
        { type: "step-start" } as Part,
        hintPart([{ id: "pm-1", content: "typed while the last turn ran" }]),
        toolPart("read"),
      ]),
    ];

    expect(splitMessagesAtDrainHints(messages)).toBe(messages);
  });

  it("keeps an empty tail segment when the hint is the newest part", () => {
    const rows = splitMessagesAtDrainHints([
      PROMPT,
      assistant("a1", [
        toolPart("read"),
        hintPart([{ id: "pm-1", content: "and book it" }]),
      ]),
    ]);

    expect(rows.map((row) => row.id)).toEqual([
      "user-1",
      "a1#seg0",
      "midturn-pm-1",
      "a1",
    ]);
    // Still the live assistant row, so the thinking indicator has a home.
    expect(rows[3].parts).toEqual([]);
  });

  it("leaves non-assistant messages and their parts untouched", () => {
    const messages = [
      PROMPT,
      assistant("a1", [
        toolPart("read"),
        hintPart([{ id: "pm-1", content: "one" }]),
        textPart("done"),
      ]),
    ];
    const rows = splitMessagesAtDrainHints(messages);

    expect(rows[0]).toBe(messages[0]);
    expect(messages[1].parts).toHaveLength(3);
  });
});
