import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import {
  ORIGINAL_TITLE,
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
