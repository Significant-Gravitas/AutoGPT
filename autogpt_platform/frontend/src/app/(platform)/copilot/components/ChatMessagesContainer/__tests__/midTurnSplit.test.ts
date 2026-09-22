import type { UIDataTypes, UIMessage, UITools } from "ai";
import { describe, expect, it } from "vitest";

import { makePromotedUserBubble } from "../../../helpers/makePromotedBubble";
import {
  asMidTurnFallbackRow,
  isMidTurnFallbackRow,
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

function statusPart(message: string): Part {
  return { type: "data-status", data: { message } } as unknown as Part;
}

function assistant(id: string, parts: Part[]): ChatMessage {
  return { id, role: "assistant", parts };
}

function user(id: string, text: string): ChatMessage {
  return { id, role: "user", parts: [textPart(text)] };
}

const PROMPT = user("user-1", "plan my week");

/** The bubble `useCopilotPendingChips` promotes above the live assistant. */
function fallback(id: string, text: string): ChatMessage {
  return makePromotedUserBubble(text, "midturn", id);
}

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

  it("does not count a text-less hint as the turn's first drawn content", () => {
    // A count-only hint (older backend, or a turn-start drain) renders
    // nothing, so a text-bearing hint right after it is still a turn-start
    // drain: splitting there would cut a segment holding only the hint.
    const messages = [
      PROMPT,
      assistant("a1", [
        hintPart(),
        hintPart([{ id: "pm-1", content: "also check Friday" }]),
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

  it("hides the fallback row whose text a hint draws at the drain point", () => {
    // The backstop GET resolved before the hint reached the client, so the
    // chip was promoted above the assistant; the hint then draws the same
    // follow-up where it belongs. One bubble, not two.
    const rows = splitMessagesAtDrainHints([
      PROMPT,
      fallback("chip-1", "follow up"),
      assistant("a1", [
        toolPart("read"),
        hintPart([{ id: "pm-1", content: "follow up" }]),
        toolPart("write"),
      ]),
    ]);

    expect(rows.map((row) => row.id)).toEqual([
      "user-1",
      "a1#seg0",
      "midturn-pm-1",
      "a1",
    ]);
  });

  it("keeps a repeated identical fallback row the hint did not draw", () => {
    const rows = splitMessagesAtDrainHints([
      PROMPT,
      fallback("chip-1", "continue"),
      fallback("chip-2", "continue"),
      assistant("a1", [
        toolPart("read"),
        hintPart([{ id: "pm-1", content: "continue" }]),
        toolPart("write"),
      ]),
    ]);

    // Two distinct drains of the same text: one drawn by its hint, the
    // other (count-only or dropped hint) still on its fallback row.
    expect(rows.map((row) => row.id)).toEqual([
      "user-1",
      "promoted-midturn-chip-2",
      "a1#seg0",
      "midturn-pm-1",
      "a1",
    ]);
  });

  it("keeps a fallback row with text no hint drew", () => {
    const rows = splitMessagesAtDrainHints([
      PROMPT,
      fallback("chip-1", "something else"),
      assistant("a1", [
        toolPart("read"),
        hintPart([{ id: "pm-1", content: "follow up" }]),
      ]),
    ]);

    expect(rows.map((row) => row.id)).toEqual([
      "user-1",
      "promoted-midturn-chip-1",
      "a1#seg0",
      "midturn-pm-1",
      "a1",
    ]);
  });

  it("only reconciles fallback rows sitting directly above the hint's turn", () => {
    const rows = splitMessagesAtDrainHints([
      PROMPT,
      fallback("chip-1", "continue"),
      assistant("a0", [textPart("earlier turn")]),
      user("user-2", "next"),
      assistant("a1", [
        toolPart("read"),
        hintPart([{ id: "pm-1", content: "continue" }]),
      ]),
    ]);

    expect(rows.map((row) => row.id)).toEqual([
      "user-1",
      "promoted-midturn-chip-1",
      "a0",
      "user-2",
      "a1#seg0",
      "midturn-pm-1",
      "a1",
    ]);
  });

  it("reaches a fallback row past the placeholder the stream leaves above the live assistant", () => {
    // The backend emits `data-status` before `start`, so `useChat` parks
    // those parts in a placeholder under its own id and pushes the real
    // message after it. The placeholder draws nothing; it must not shield
    // the fallback row above it from the bubble the hint draws.
    const rows = splitMessagesAtDrainHints([
      PROMPT,
      fallback("pending-chip-local-1", "also check Friday"),
      assistant("placeholder", [statusPart("Preparing workspace…")]),
      assistant("a1", [
        toolPart("read"),
        hintPart([{ id: "pm-1", content: "also check Friday" }]),
        toolPart("write"),
      ]),
    ]);

    expect(rows.map((row) => row.id)).toEqual([
      "user-1",
      "placeholder",
      "a1#seg0",
      "midturn-pm-1",
      "a1",
    ]);
  });

  it("drops a bubble the auto-continue path promoted once the hint draws it", () => {
    // The SDK swapping its placeholder id for the server's message id looks
    // like an auto-continue to `useCopilotPendingChips`, which then promotes
    // the mid-turn chip under that flavour. Same bubble, same reconciliation.
    const rows = splitMessagesAtDrainHints([
      PROMPT,
      makePromotedUserBubble(
        "also check Friday",
        "auto-continue",
        "pending-chip-local-1",
      ),
      assistant("a1", [
        toolPart("read"),
        hintPart([{ id: "pm-1", content: "also check Friday" }]),
        toolPart("write"),
      ]),
    ]);

    expect(rows.map((row) => row.id)).toEqual([
      "user-1",
      "a1#seg0",
      "midturn-pm-1",
      "a1",
    ]);
  });

  it("treats a hydrated mid-turn row the resume cut kept as a fallback row", () => {
    const hydrated = user("session-seq-5", "and book it");
    const kept = asMidTurnFallbackRow(hydrated);

    expect(isMidTurnFallbackRow(hydrated)).toBe(false);
    expect(isMidTurnFallbackRow(kept)).toBe(true);
    expect(isMidTurnFallbackRow(fallback("chip-1", "x"))).toBe(true);
    expect(isMidTurnFallbackRow(PROMPT)).toBe(false);
    // Idempotent, and the row keeps its parts and its sequence suffix.
    expect(asMidTurnFallbackRow(kept)).toBe(kept);
    expect(kept.parts).toBe(hydrated.parts);
    expect(kept.id.endsWith("-seq-5")).toBe(true);

    const rows = splitMessagesAtDrainHints([
      PROMPT,
      kept,
      assistant("a1", [
        toolPart("read"),
        hintPart([{ id: "pm-1", content: "and book it" }]),
        toolPart("write"),
      ]),
    ]);
    expect(rows.map((row) => row.id)).toEqual([
      "user-1",
      "a1#seg0",
      "midturn-pm-1",
      "a1",
    ]);
  });
});
