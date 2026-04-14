import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import {
  ORIGINAL_TITLE,
  deduplicateMessages,
  extractSendMessageText,
  formatNotificationTitle,
  getSendSuppressionReason,
  parseSessionIDs,
  shouldSuppressDuplicateSend,
} from "./helpers";

describe("formatNotificationTitle", () => {
  it("returns base title when count is 0", () => {
    expect(formatNotificationTitle(0)).toBe(ORIGINAL_TITLE);
  });

  it("returns formatted title with count", () => {
    expect(formatNotificationTitle(3)).toBe(
      `(3) AutoPilot is ready - ${ORIGINAL_TITLE}`,
    );
  });

  it("returns base title for negative count", () => {
    expect(formatNotificationTitle(-1)).toBe(ORIGINAL_TITLE);
  });

  it("returns base title for NaN", () => {
    expect(formatNotificationTitle(NaN)).toBe(ORIGINAL_TITLE);
  });

  it("returns formatted title for count of 1", () => {
    expect(formatNotificationTitle(1)).toBe(
      `(1) AutoPilot is ready - ${ORIGINAL_TITLE}`,
    );
  });
});

describe("parseSessionIDs", () => {
  it("returns empty set for null", () => {
    expect(parseSessionIDs(null)).toEqual(new Set());
  });

  it("returns empty set for undefined", () => {
    expect(parseSessionIDs(undefined)).toEqual(new Set());
  });

  it("returns empty set for empty string", () => {
    expect(parseSessionIDs("")).toEqual(new Set());
  });

  it("parses valid JSON array of strings", () => {
    expect(parseSessionIDs('["a","b","c"]')).toEqual(new Set(["a", "b", "c"]));
  });

  it("filters out non-string elements", () => {
    expect(parseSessionIDs('[1,"valid",null,true,"also-valid"]')).toEqual(
      new Set(["valid", "also-valid"]),
    );
  });

  it("returns empty set for non-array JSON", () => {
    expect(parseSessionIDs('{"key":"value"}')).toEqual(new Set());
  });

  it("returns empty set for JSON string value", () => {
    expect(parseSessionIDs('"oops"')).toEqual(new Set());
  });

  it("returns empty set for JSON number value", () => {
    expect(parseSessionIDs("42")).toEqual(new Set());
  });

  it("returns empty set for malformed JSON", () => {
    expect(parseSessionIDs("{broken")).toEqual(new Set());
  });

  it("deduplicates entries", () => {
    expect(parseSessionIDs('["a","a","b"]')).toEqual(new Set(["a", "b"]));
  });
});

describe("extractSendMessageText", () => {
  it("extracts text from a string argument", () => {
    expect(extractSendMessageText("hello")).toBe("hello");
  });

  it("extracts text from an object with text property", () => {
    expect(extractSendMessageText({ text: "world" })).toBe("world");
  });

  it("returns empty string for null", () => {
    expect(extractSendMessageText(null)).toBe("");
  });

  it("returns empty string for undefined", () => {
    expect(extractSendMessageText(undefined)).toBe("");
  });

  it("converts numbers to string", () => {
    expect(extractSendMessageText(42)).toBe("42");
  });
});

let msgCounter = 0;
function makeMsg(role: "user" | "assistant", text: string): UIMessage {
  return {
    id: `msg-${msgCounter++}`,
    role,
    parts: [{ type: "text", text }],
  };
}

describe("shouldSuppressDuplicateSend", () => {
  it("suppresses when reconnect is scheduled", () => {
    expect(
      shouldSuppressDuplicateSend({
        text: "hello",
        isReconnectScheduled: true,
        lastSubmittedText: null,
        messages: [],
      }),
    ).toBe(true);
  });

  it("allows send when not reconnecting and no prior submission", () => {
    expect(
      shouldSuppressDuplicateSend({
        text: "hello",
        isReconnectScheduled: false,
        lastSubmittedText: null,
        messages: [],
      }),
    ).toBe(false);
  });

  it("suppresses when text matches last submitted AND last user message", () => {
    const messages = [makeMsg("user", "hello"), makeMsg("assistant", "hi")];
    expect(
      shouldSuppressDuplicateSend({
        text: "hello",
        isReconnectScheduled: false,
        lastSubmittedText: "hello",
        messages,
      }),
    ).toBe(true);
  });

  it("allows send when text matches last submitted but differs from last user message", () => {
    const messages = [
      makeMsg("user", "different"),
      makeMsg("assistant", "reply"),
    ];
    expect(
      shouldSuppressDuplicateSend({
        text: "hello",
        isReconnectScheduled: false,
        lastSubmittedText: "hello",
        messages,
      }),
    ).toBe(false);
  });

  it("allows send when text differs from last submitted", () => {
    const messages = [makeMsg("user", "hello")];
    expect(
      shouldSuppressDuplicateSend({
        text: "new message",
        isReconnectScheduled: false,
        lastSubmittedText: "hello",
        messages,
      }),
    ).toBe(false);
  });

  it("allows send when text is empty", () => {
    expect(
      shouldSuppressDuplicateSend({
        text: "",
        isReconnectScheduled: false,
        lastSubmittedText: "",
        messages: [],
      }),
    ).toBe(false);
  });

  it("allows send with empty messages array even if text matches lastSubmitted", () => {
    expect(
      shouldSuppressDuplicateSend({
        text: "hello",
        isReconnectScheduled: false,
        lastSubmittedText: "hello",
        messages: [],
      }),
    ).toBe(false);
  });
});

describe("getSendSuppressionReason", () => {
  it("returns 'reconnecting' when reconnect is scheduled", () => {
    expect(
      getSendSuppressionReason({
        text: "hello",
        isReconnectScheduled: true,
        lastSubmittedText: null,
        messages: [],
      }),
    ).toBe("reconnecting");
  });

  it("returns 'reconnecting' even when text would otherwise be a duplicate", () => {
    const messages = [makeMsg("user", "hello")];
    expect(
      getSendSuppressionReason({
        text: "hello",
        isReconnectScheduled: true,
        lastSubmittedText: "hello",
        messages,
      }),
    ).toBe("reconnecting");
  });

  it("returns 'duplicate' when text matches last submitted AND last user message", () => {
    const messages = [makeMsg("user", "hello"), makeMsg("assistant", "hi")];
    expect(
      getSendSuppressionReason({
        text: "hello",
        isReconnectScheduled: false,
        lastSubmittedText: "hello",
        messages,
      }),
    ).toBe("duplicate");
  });

  it("returns null when text matches last submitted but differs from last user message", () => {
    const messages = [
      makeMsg("user", "different"),
      makeMsg("assistant", "reply"),
    ];
    expect(
      getSendSuppressionReason({
        text: "hello",
        isReconnectScheduled: false,
        lastSubmittedText: "hello",
        messages,
      }),
    ).toBeNull();
  });

  it("returns null when text differs from last submitted", () => {
    const messages = [makeMsg("user", "hello")];
    expect(
      getSendSuppressionReason({
        text: "new message",
        isReconnectScheduled: false,
        lastSubmittedText: "hello",
        messages,
      }),
    ).toBeNull();
  });

  it("returns null when not reconnecting and no prior submission", () => {
    expect(
      getSendSuppressionReason({
        text: "hello",
        isReconnectScheduled: false,
        lastSubmittedText: null,
        messages: [],
      }),
    ).toBeNull();
  });

  it("returns null when text is empty", () => {
    expect(
      getSendSuppressionReason({
        text: "",
        isReconnectScheduled: false,
        lastSubmittedText: "",
        messages: [],
      }),
    ).toBeNull();
  });

  it("returns null when messages array is empty even if text matches lastSubmitted", () => {
    expect(
      getSendSuppressionReason({
        text: "hello",
        isReconnectScheduled: false,
        lastSubmittedText: "hello",
        messages: [],
      }),
    ).toBeNull();
  });
});

// Helper that creates messages with explicit IDs for dedup tests
function makeMsgWithId(
  id: string,
  role: "user" | "assistant",
  text: string,
): UIMessage {
  return { id, role, parts: [{ type: "text", text }] };
}

describe("deduplicateMessages", () => {
  it("removes messages with duplicate IDs", () => {
    const msgs = [
      makeMsgWithId("1", "user", "hello"),
      makeMsgWithId("1", "user", "hello"),
    ];
    expect(deduplicateMessages(msgs)).toHaveLength(1);
  });

  it("removes non-adjacent assistant duplicates with different IDs (SSE replay)", () => {
    const msgs = [
      makeMsgWithId("u1", "user", "hello"),
      makeMsgWithId("a1", "assistant", "Plan of Attack"),
      makeMsgWithId("a2", "assistant", "Next step"),
      // SSE replay appends the same content with new IDs
      makeMsgWithId("a3", "assistant", "Plan of Attack"),
      makeMsgWithId("a4", "assistant", "Next step"),
    ];
    const result = deduplicateMessages(msgs);
    expect(result).toHaveLength(3); // user + 2 unique assistant msgs
    expect(result.map((m) => m.id)).toEqual(["u1", "a1", "a2"]);
  });

  it("keeps identical assistant replies to different user prompts", () => {
    const msgs = [
      makeMsgWithId("u1", "user", "What is 2+2?"),
      makeMsgWithId("a1", "assistant", "4"),
      makeMsgWithId("u2", "user", "What is 1+3?"),
      makeMsgWithId("a2", "assistant", "4"),
    ];
    const result = deduplicateMessages(msgs);
    expect(result).toHaveLength(4);
  });

  it("keeps second answer when same question is asked twice in one session", () => {
    // Regression: scoping by user message TEXT instead of ID would treat both
    // turns as the same context and drop the second identical assistant reply.
    const msgs = [
      makeMsgWithId("u1", "user", "What is 2+2?"),
      makeMsgWithId("a1", "assistant", "4"),
      makeMsgWithId("u2", "user", "What is 2+2?"), // same question, different ID
      makeMsgWithId("a2", "assistant", "4"), // same answer — must be kept
    ];
    const result = deduplicateMessages(msgs);
    expect(result).toHaveLength(4);
    expect(result.map((m) => m.id)).toEqual(["u1", "a1", "u2", "a2"]);
  });

  it("removes adjacent assistant duplicates", () => {
    const msgs = [
      makeMsgWithId("u1", "user", "hello"),
      makeMsgWithId("a1", "assistant", "hi there"),
      makeMsgWithId("a2", "assistant", "hi there"),
    ];
    const result = deduplicateMessages(msgs);
    expect(result).toHaveLength(2);
  });

  it("handles empty message list", () => {
    expect(deduplicateMessages([])).toEqual([]);
  });

  it("passes through unique messages unchanged", () => {
    const msgs = [
      makeMsgWithId("u1", "user", "question 1"),
      makeMsgWithId("a1", "assistant", "answer 1"),
      makeMsgWithId("u2", "user", "question 2"),
      makeMsgWithId("a2", "assistant", "answer 2"),
    ];
    expect(deduplicateMessages(msgs)).toHaveLength(4);
  });

  it("does not create false positives for text parts that contain the separator", () => {
    // "a|b" + "c" and "a" + "b|c" previously collided when joined with "|"
    const msgs: UIMessage[] = [
      makeMsgWithId("u1", "user", "hello"),
      {
        id: "a1",
        role: "assistant",
        parts: [
          { type: "text", text: "a|b" },
          { type: "text", text: "c" },
        ],
      },
      {
        id: "a2",
        role: "assistant",
        parts: [
          { type: "text", text: "a" },
          { type: "text", text: "b|c" },
        ],
      },
    ];
    const result = deduplicateMessages(msgs);
    expect(result).toHaveLength(3); // both assistant messages should be kept
  });

  it("deduplicates by toolCallId for tool-call parts", () => {
    const msgs: UIMessage[] = [
      makeMsgWithId("u1", "user", "run tool"),
      {
        id: "a1",
        role: "assistant",
        parts: [
          {
            type: "dynamic-tool",
            toolCallId: "tc-1",
            toolName: "test",
            state: "input-available",
            input: {},
          },
        ],
      },
      {
        id: "a2",
        role: "assistant",
        parts: [
          {
            type: "dynamic-tool",
            toolCallId: "tc-1",
            toolName: "test",
            state: "input-available",
            input: {},
          },
        ],
      },
    ];
    const result = deduplicateMessages(msgs);
    expect(result).toHaveLength(2); // user + first tool call
  });

  it("passes through assistant messages with empty parts without deduplicating them", () => {
    // contentFingerprint === "[]" when parts is empty; the guard skips fingerprint
    // tracking so these messages are never incorrectly deduplicated against each other.
    const msgs: UIMessage[] = [
      makeMsgWithId("u1", "user", "hello"),
      { id: "a1", role: "assistant", parts: [] },
      { id: "a2", role: "assistant", parts: [] },
    ];
    const result = deduplicateMessages(msgs);
    expect(result).toHaveLength(3); // both empty-parts messages are kept
  });

  it("does not collapse structurally different no-text parts to the same fingerprint", () => {
    // Parts lacking both 'text' and 'toolCallId' (e.g. step-start) previously
    // all mapped to "" causing false-positive deduplication. Now JSON.stringify(p)
    // is used as the fallback so distinct part shapes produce distinct fingerprints.
    const msgs: UIMessage[] = [
      makeMsgWithId("u1", "user", "hello"),
      {
        id: "a1",
        role: "assistant",
        parts: [{ type: "step-start" }],
      },
      {
        id: "a2",
        role: "assistant",
        parts: [{ type: "step-start" }],
      },
    ];
    const result = deduplicateMessages(msgs);
    expect(result).toHaveLength(2); // duplicate step-start messages are deduped
  });
});
