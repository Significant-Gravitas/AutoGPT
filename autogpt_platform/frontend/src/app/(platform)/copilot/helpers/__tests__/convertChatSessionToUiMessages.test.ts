import type { UIDataTypes, UIMessage, UITools } from "ai";
import { describe, expect, it } from "vitest";
import {
  concatWithAssistantMerge,
  convertChatSessionMessagesToUiMessages,
} from "../convertChatSessionToUiMessages";

const SESSION_ID = "sess-test";

describe("convertChatSessionMessagesToUiMessages", () => {
  it("does not drop user messages with null content", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [{ role: "user", content: null, sequence: 0 }],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].role).toBe("user");
  });

  it("does not drop user messages with empty string content", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [{ role: "user", content: "", sequence: 0 }],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].role).toBe("user");
  });

  it("still drops non-user messages with null content", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [{ role: "assistant", content: null, sequence: 0 }],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(0);
  });

  it("still drops non-user messages with empty string content", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [{ role: "assistant", content: "", sequence: 0 }],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(0);
  });

  it("includes user message with normal content", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [{ role: "user", content: "hello", sequence: 0 }],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].role).toBe("user");
  });

  it("attaches a reasoning row between user/assistant to the surrounding assistant bubble", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "user", content: "hi", sequence: 0 },
        { role: "reasoning", content: "thinking deeply", sequence: 1 },
        { role: "assistant", content: "hello back", sequence: 2 },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(2);
    expect(result.messages[0].role).toBe("user");
    const assistant = result.messages[1];
    expect(assistant.role).toBe("assistant");
    expect(assistant.parts).toHaveLength(2);
    expect(assistant.parts[0]).toMatchObject({
      type: "reasoning",
      text: "thinking deeply",
      state: "done",
    });
    expect(assistant.parts[1]).toMatchObject({
      type: "text",
      text: "hello back",
    });
  });

  it("merges separate reasoning and assistant DB rows into one UIMessage with reasoning first", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "reasoning", content: "plan", sequence: 0 },
        { role: "assistant", content: "answer", sequence: 1 },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].role).toBe("assistant");
    expect(result.messages[0].parts.map((p) => p.type)).toEqual([
      "reasoning",
      "text",
    ]);
  });

  it("includes every reasoning row when multiple are present in the same turn", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "reasoning", content: "step one", sequence: 0 },
        { role: "reasoning", content: "step two", sequence: 1 },
        { role: "assistant", content: "done", sequence: 2 },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    const parts = result.messages[0].parts;
    const reasoningParts = parts.filter((p) => p.type === "reasoning");
    expect(reasoningParts).toHaveLength(2);
    expect(reasoningParts[0]).toMatchObject({ text: "step one" });
    expect(reasoningParts[1]).toMatchObject({ text: "step two" });
  });

  it("skips reasoning rows with empty content so no reasoning part is emitted", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "reasoning", content: "", sequence: 0 },
        { role: "assistant", content: "answer", sequence: 1 },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    const parts = result.messages[0].parts;
    expect(parts.some((p) => p.type === "reasoning")).toBe(false);
    expect(parts[0]).toMatchObject({ type: "text", text: "answer" });
  });

  it("captures duration_ms from the following assistant row on the merged bubble", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "reasoning", content: "ponder", sequence: 0 },
        { role: "assistant", content: "reply", sequence: 1, duration_ms: 750 },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    const mergedId = result.messages[0].id;
    expect(result.stats.get(mergedId)?.durationMs).toBe(750);
  });

  it("falls back to idx-based ids when sequence is null so sequence-less rows don't collide", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "user", content: "first" },
        { role: "assistant", content: "reply one" },
        { role: "user", content: "second" },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(3);
    const ids = result.messages.map((m) => m.id);
    expect(new Set(ids).size).toBe(3);
    for (const id of ids) {
      expect(id.startsWith(`${SESSION_ID}-idx-`)).toBe(true);
    }
  });

  it("uses sequence-based id when sequence is present and idx-based otherwise in the same list", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "user", content: "seq-ed", sequence: 7 },
        { role: "assistant", content: "no-seq reply" },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(2);
    expect(result.messages[0].id).toBe(`${SESSION_ID}-seq-7`);
    expect(result.messages[1].id).toBe(`${SESSION_ID}-idx-1`);
  });

  it("skips role values that are neither user, assistant, tool, nor reasoning", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "system", content: "ignored", sequence: 0 },
        { role: "assistant", content: "kept", sequence: 1 },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].role).toBe("assistant");
  });

  it("captures duration_ms directly on a standalone assistant row (non-merged branch)", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "user", content: "hi", sequence: 0 },
        { role: "assistant", content: "reply", sequence: 1, duration_ms: 123 },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(2);
    const assistantId = result.messages[1].id;
    expect(result.stats.get(assistantId)?.durationMs).toBe(123);
  });

  it("captures created_at when supplied as an ISO string", () => {
    const iso = "2026-04-23T01:32:09.871Z";
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [{ role: "user", content: "hi", sequence: 0, created_at: iso }],
      { isComplete: true },
    );

    const userId = result.messages[0].id;
    expect(result.stats.get(userId)?.createdAt).toBe(iso);
  });

  it("captures created_at when the API mutator has already converted the field to a Date object", () => {
    // The generated `customMutator` runs `transformDates()` on every response,
    // which turns ISO date strings into Date objects before they reach the
    // UI-shape converter.  A literal `typeof === "string"` check would reject
    // the Date and silently drop the timestamp — breaking the "Thought for X"
    // tooltip.  Assert we still recover the ISO value.
    const date = new Date("2026-04-23T01:32:09.871Z");
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [{ role: "user", content: "hi", sequence: 0, created_at: date }],
      { isComplete: true },
    );

    const userId = result.messages[0].id;
    expect(result.stats.get(userId)?.createdAt).toBe(date.toISOString());
  });

  it("advances createdAt to the latest row when merging consecutive assistant rows", () => {
    // Reasoning row persisted early + assistant row persisted later should
    // leave the merged bubble's stats.createdAt pointing at the LATER row,
    // so the live "Thinking Xs" counter anchors to the most recent step.
    const early = "2026-04-23T10:00:00.000Z";
    const later = "2026-04-23T10:00:30.000Z";
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "user", content: "hi", sequence: 0, created_at: early },
        {
          role: "reasoning",
          content: "ponder",
          sequence: 1,
          created_at: early,
        },
        {
          role: "assistant",
          content: "reply",
          sequence: 2,
          created_at: later,
        },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(2);
    const mergedId = result.messages[1].id;
    expect(result.stats.get(mergedId)?.createdAt).toBe(later);
  });
});

// --------------------------------------------------------------------------- //
//  concatWithAssistantMerge — page-boundary stitching                         //
// --------------------------------------------------------------------------- //

function uiAssistant(
  sessionId: string,
  seq: number,
  text: string,
): UIMessage<unknown, UIDataTypes, UITools> {
  return {
    id: `${sessionId}-seq-${seq}`,
    role: "assistant",
    parts: [{ type: "text", text, state: "done" }],
  };
}

function uiUser(
  sessionId: string,
  seq: number,
  text: string,
): UIMessage<unknown, UIDataTypes, UITools> {
  return {
    id: `${sessionId}-seq-${seq}`,
    role: "user",
    parts: [{ type: "text", text, state: "done" }],
  };
}

describe("concatWithAssistantMerge", () => {
  it("returns b unchanged when a is empty", () => {
    const b = [uiAssistant(SESSION_ID, 0, "hi")];
    expect(concatWithAssistantMerge([], b)).toEqual(b);
  });

  it("returns a unchanged when b is empty", () => {
    const a = [uiAssistant(SESSION_ID, 0, "hi")];
    expect(concatWithAssistantMerge(a, [])).toEqual(a);
  });

  it("merges two ASSISTANT bubbles whose DB sequences are strictly adjacent", () => {
    const a = [uiAssistant(SESSION_ID, 2, "hello")];
    const b = [uiAssistant(SESSION_ID, 3, " world")];
    const result = concatWithAssistantMerge(a, b);
    expect(result).toHaveLength(1);
    expect(result[0].role).toBe("assistant");
    expect(result[0].parts).toHaveLength(2);
  });

  it("does NOT merge when DB sequences are not adjacent — a hidden user/reasoning row would be silently swallowed", () => {
    // Real-world repro: assistant_seq3 + (user_seq4 not yet hydrated) +
    // assistant_seq6.  Pre-fix this stitched seq3 and seq6 into one bubble
    // and the chip's user follow-up at seq4 vanished.
    const a = [uiAssistant(SESSION_ID, 3, "before")];
    const b = [uiAssistant(SESSION_ID, 6, "after")];
    const result = concatWithAssistantMerge(a, b);
    expect(result).toHaveLength(2);
    expect(result[0].id).toBe(`${SESSION_ID}-seq-3`);
    expect(result[1].id).toBe(`${SESSION_ID}-seq-6`);
  });

  it("does NOT merge when last-of-a is user and first-of-b is assistant", () => {
    const a = [uiUser(SESSION_ID, 4, "follow up")];
    const b = [uiAssistant(SESSION_ID, 5, "got it")];
    const result = concatWithAssistantMerge(a, b);
    expect(result).toHaveLength(2);
  });

  it("merges a single-row page B onto a multi-row merged bubble in page A using the LAST seq of the bubble", () => {
    // Sentry: pre-fix, a merged bubble holding seq=5+6 was keyed
    // ``-seq-5``, and a cross-page assistant at seq=7 failed the
    // ``firstSeq === lastSeq + 1`` check (7 !== 5+1) and split into
    // two bubbles.  The in-page merge now advances the id to the last
    // seq, so the adjacency check sees ``7 === 6+1`` and merges.
    const pageA = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "reasoning", content: "thinking", sequence: 5 },
        { role: "assistant", content: "first part", sequence: 6 },
      ],
      { isComplete: true },
    ).messages;
    expect(pageA).toHaveLength(1);
    expect(pageA[0].id).toBe(`${SESSION_ID}-seq-6`);
    const pageB = [uiAssistant(SESSION_ID, 7, "continued")];
    const result = concatWithAssistantMerge(pageA, pageB);
    expect(result).toHaveLength(1);
    expect(result[0].parts.length).toBeGreaterThan(1);
  });

  it("does NOT merge across hydrated → streaming boundary (streaming ids fail seq-extraction)", () => {
    const a = [uiAssistant(SESSION_ID, 5, "from db")];
    const b: UIMessage<unknown, UIDataTypes, UITools>[] = [
      {
        id: "ai-sdk-streaming-uuid",
        role: "assistant",
        parts: [{ type: "text", text: " streaming", state: "streaming" }],
      },
    ];
    const result = concatWithAssistantMerge(a, b);
    // Refusing to merge is the safer default — streaming consumer handles
    // its own assistant continuity inside the active turn.
    expect(result).toHaveLength(2);
  });
});

describe("convertChatSessionMessagesToUiMessages — latest user marker", () => {
  it("marks the latest user row as isLatestUserMessage=true", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { id: "uuid-user-1", role: "user", content: "first", sequence: 0 },
        { id: "uuid-asst-1", role: "assistant", content: "reply", sequence: 1 },
        { id: "uuid-user-2", role: "user", content: "newest", sequence: 2 },
      ],
      { isComplete: true },
    );

    const stats0 = result.stats.get(result.messages[0].id);
    const stats2 = result.stats.get(
      result.messages[result.messages.length - 1].id,
    );
    // First user row is older — not the latest.
    expect(stats0?.isLatestUserMessage).toBeFalsy();
    // Latest user row carries the marker; consumer (ChatMessagesContainer)
    // decides whether to render the Queued badge by also checking
    // session.chat_status === 'queued'.
    expect(stats2?.isLatestUserMessage).toBe(true);
    expect(stats2?.rawMessageId).toBe("uuid-user-2");
  });

  it("marks a single user row as the latest", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [{ id: "u1", role: "user", content: "hi", sequence: 0 }],
      { isComplete: true },
    );

    const stats = result.stats.get(result.messages[0].id);
    expect(stats?.isLatestUserMessage).toBe(true);
  });
});
