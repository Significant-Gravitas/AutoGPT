import type { UIMessage } from "ai";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
  type MockInstance,
} from "vitest";
import { IMPERSONATION_HEADER_NAME } from "@/lib/constants";
import {
  COPILOT_COMPLETION_NOTIFICATION,
  deduplicateMessages,
  extractSendMessageText,
  getCopilotAuthHeaders,
  extractSendMessageAttachments,
  getSendSuppressionReason,
  hasVisibleAssistantContent,
  parseSessionIDs,
  isEngineSwitchPart,
  resolveSessionDryRun,
  shouldDebounceReconnect,
  shouldSuppressDuplicateSend,
} from "../helpers";

vi.mock("@/lib/auth/actions", () => ({
  getWebSocketToken: vi.fn(),
}));

vi.mock("@/lib/impersonation", () => ({
  getSystemHeaders: vi.fn(),
}));

vi.mock("@/services/analytics/posthog-request-context", () => ({
  getPostHogSessionHeader: vi.fn(() => ({})),
}));

import { getWebSocketToken } from "@/lib/auth/actions";
import { getSystemHeaders } from "@/lib/impersonation";
import { getPostHogSessionHeader } from "@/services/analytics/posthog-request-context";

const mockGetWebSocketToken = vi.mocked(getWebSocketToken);
const mockGetSystemHeaders = vi.mocked(getSystemHeaders);

describe("hasVisibleAssistantContent", () => {
  function assistant(parts: UIMessage["parts"]): UIMessage[] {
    return [{ id: "a1", role: "assistant", parts }];
  }

  it("counts a Paused card, the only thing a turn refused before it ran shows", () => {
    expect(
      hasVisibleAssistantContent(
        assistant([
          { type: "data-status", data: { message: "Picking it up…" } },
          { type: "data-turn-paused", data: { reason: "daily" } },
        ]),
      ),
    ).toBe(true);
  });

  it("does not count bookkeeping alone", () => {
    expect(
      hasVisibleAssistantContent(
        assistant([
          { type: "step-start" },
          { type: "data-status", data: { message: "Picking it up…" } },
        ]),
      ),
    ).toBe(false);
  });
});

describe("COPILOT_COMPLETION_NOTIFICATION", () => {
  it("matches the copy hardcoded in public/push-sw.js", () => {
    expect(COPILOT_COMPLETION_NOTIFICATION).toEqual({
      title: "AutoGPT",
      body: "Task completed",
      icon: "/notification-icon-192.png",
    });
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

  it("suppresses when text matches an unanswered last user message", () => {
    const messages = [makeMsg("user", "hello")];
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
  const failedMessages = [
    makeMsg("user", "hello"),
    makeMsg(
      "assistant",
      "[__COPILOT_RETRYABLE_ERROR_a9c2__] The model returned an empty response.",
    ),
  ];

  it.each(["ready", "error"] as const)(
    "allows retrying a failed turn when status is %s",
    (status) => {
      expect(
        getSendSuppressionReason({
          text: "hello",
          isReconnectScheduled: false,
          lastSubmittedText: "hello",
          messages: failedMessages,
          status,
        }),
      ).toBeNull();
    },
  );

  it("allows resending after an SDK error without a persisted marker", () => {
    expect(
      getSendSuppressionReason({
        text: "hello",
        isReconnectScheduled: false,
        lastSubmittedText: "hello",
        messages: [makeMsg("user", "hello")],
        status: "error",
      }),
    ).toBeNull();
  });

  it.each(["streaming", "submitted"] as const)(
    "still suppresses duplicates while status is %s",
    (status) => {
      expect(
        getSendSuppressionReason({
          text: "hello",
          isReconnectScheduled: false,
          lastSubmittedText: "hello",
          messages: failedMessages,
          status,
        }),
      ).toBe("duplicate");
    },
  );

  it("still suppresses a retry during reconnect", () => {
    expect(
      getSendSuppressionReason({
        text: "hello",
        isReconnectScheduled: true,
        lastSubmittedText: "hello",
        messages: failedMessages,
        status: "error",
      }),
    ).toBe("reconnecting");
  });

  it("does not treat an earlier turn's error marker as a failed current turn", () => {
    expect(
      getSendSuppressionReason({
        text: "hello",
        isReconnectScheduled: false,
        lastSubmittedText: "hello",
        messages: [...failedMessages, makeMsg("user", "hello")],
        status: "ready",
      }),
    ).toBe("duplicate");
  });

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

  it("returns 'duplicate' while the echoed send is still unanswered", () => {
    const messages = [makeMsg("user", "hello")];
    expect(
      getSendSuppressionReason({
        text: "hello",
        isReconnectScheduled: false,
        lastSubmittedText: "hello",
        messages,
        status: "ready",
      }),
    ).toBe("duplicate");
  });

  it("sends the same text again once it has been answered", () => {
    const messages = [makeMsg("user", "hello"), makeMsg("assistant", "hi")];
    expect(
      getSendSuppressionReason({
        text: "hello",
        isReconnectScheduled: false,
        lastSubmittedText: "hello",
        messages,
        status: "ready",
      }),
    ).toBeNull();
  });

  it("sends the same text again after the previous turn was stopped", () => {
    const messages = [
      makeMsg("user", "hello"),
      makeMsg("assistant", "[__COPILOT_ERROR_f7a1__] Operation cancelled"),
    ];
    expect(
      getSendSuppressionReason({
        text: "hello",
        isReconnectScheduled: false,
        lastSubmittedText: "hello",
        messages,
        status: "ready",
      }),
    ).toBeNull();
  });

  it("lets the same caption through when it carries other attachments", () => {
    const messages = [makeMsg("user", "summarize this")];
    expect(
      getSendSuppressionReason({
        text: "summarize this",
        attachments: extractSendMessageAttachments({
          text: "summarize this",
          files: [{ type: "file", mediaType: "image/png", url: "/f/new" }],
        }),
        isReconnectScheduled: false,
        lastSubmittedText: "summarize this",
        messages,
      }),
    ).toBeNull();
  });

  it("still drops a resend of the same caption with the same attachments", () => {
    const file = { type: "file" as const, mediaType: "image/png", url: "/f/1" };
    const messages: UIMessage[] = [
      {
        id: "msg-with-file",
        role: "user",
        parts: [file, { type: "text", text: "summarize this" }],
      },
    ];
    expect(
      getSendSuppressionReason({
        text: "summarize this",
        attachments: extractSendMessageAttachments({
          text: "summarize this",
          files: [file],
        }),
        isReconnectScheduled: false,
        lastSubmittedText: "summarize this",
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
    // Parts lacking both 'text' and 'toolCallId' (e.g. step-start, step-end)
    // previously all mapped to "" causing false-positive deduplication. Now
    // JSON.stringify(p) is the fallback so distinct part shapes produce
    // distinct fingerprints — and identically-shaped no-text parts still dedupe.
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
        parts: [{ type: "step-end" } as unknown as UIMessage["parts"][number]],
      },
      {
        id: "a3",
        role: "assistant",
        parts: [{ type: "step-start" }],
      },
    ];
    const result = deduplicateMessages(msgs);
    // user + step-start + step-end kept; the second step-start dedupes against a1.
    expect(result).toHaveLength(3);
    expect(result.map((m) => m.id)).toEqual(["u1", "a1", "a2"]);
  });

  it("returns the input array itself when nothing is a duplicate", () => {
    const msgs = [
      makeMsgWithId("u1", "user", "question 1"),
      makeMsgWithId("a1", "assistant", "answer 1"),
      makeMsgWithId("u2", "user", "question 2"),
      makeMsgWithId("a2", "assistant", "answer 2"),
    ];
    expect(deduplicateMessages(msgs)).toBe(msgs);
    expect(deduplicateMessages([])).toEqual([]);
  });

  it("returns a new array when it removes a duplicate", () => {
    const msgs = [
      makeMsgWithId("u1", "user", "hello"),
      makeMsgWithId("a1", "assistant", "hi there"),
      makeMsgWithId("a2", "assistant", "hi there"),
    ];
    const result = deduplicateMessages(msgs);
    expect(result).not.toBe(msgs);
    expect(result.map((m) => m.id)).toEqual(["u1", "a1"]);
  });

  describe("fingerprint cache", () => {
    let stringify: MockInstance<typeof JSON.stringify>;
    beforeEach(() => {
      stringify = vi.spyOn(JSON, "stringify");
    });
    afterEach(() => {
      stringify.mockRestore();
    });

    it("fingerprints an unchanged assistant message once, and the newest one on every call", () => {
      const older = makeMsgWithId("a1", "assistant", "answer 1");
      const msgs = [
        makeMsgWithId("u1", "user", "question 1"),
        older,
        makeMsgWithId("u2", "user", "question 2"),
        makeMsgWithId("a2", "assistant", "answer 2"),
      ];

      deduplicateMessages(msgs);
      expect(stringify).toHaveBeenCalledTimes(2);

      stringify.mockClear();
      const streamed = {
        ...msgs[3],
        parts: [{ type: "text" as const, text: "answer 2, longer" }],
      };
      deduplicateMessages([...msgs.slice(0, 3), streamed]);
      expect(stringify).toHaveBeenCalledTimes(1);
      expect(stringify.mock.calls[0][0]).toEqual(["answer 2, longer"]);
    });

    it("fingerprints a message again once it gets new parts", () => {
      const msgs: UIMessage[] = [
        makeMsgWithId("u1", "user", "question"),
        makeMsgWithId("a1", "assistant", "same"),
        makeMsgWithId("a2", "assistant", "other"),
        makeMsgWithId("u2", "user", "next"),
      ];
      expect(deduplicateMessages(msgs)).toBe(msgs);

      msgs[2] = { ...msgs[2], parts: [{ type: "text", text: "same" }] };
      expect(deduplicateMessages(msgs).map((m) => m.id)).toEqual([
        "u1",
        "a1",
        "u2",
      ]);
    });

    it("fingerprints a message again when parts are added to it in place", () => {
      const growing: UIMessage = { id: "a2", role: "assistant", parts: [] };
      const msgs = [
        makeMsgWithId("u1", "user", "question"),
        makeMsgWithId("a1", "assistant", "same"),
        growing,
        makeMsgWithId("u2", "user", "next"),
      ];
      expect(deduplicateMessages(msgs)).toBe(msgs);

      growing.parts.push({ type: "text", text: "same" });
      expect(deduplicateMessages(msgs).map((m) => m.id)).toEqual([
        "u1",
        "a1",
        "u2",
      ]);
    });
  });
});

describe("shouldDebounceReconnect", () => {
  const WINDOW_MS = 1_500;

  it("returns null for the first reconnect (lastResumeAt === 0)", () => {
    expect(shouldDebounceReconnect(0, 10_000, WINDOW_MS)).toBeNull();
  });

  it("returns null for a negative lastResumeAt sentinel", () => {
    // Defensive: a negative value is still treated as "no reconnect yet".
    expect(shouldDebounceReconnect(-1, 10_000, WINDOW_MS)).toBeNull();
  });

  it("returns the remaining delay when now is inside the window", () => {
    // 500ms since the last resume — the caller must wait another 1000ms
    // before the storm cap reopens.
    const remaining = shouldDebounceReconnect(1_000, 1_500, WINDOW_MS);
    expect(remaining).toBe(1_000);
  });

  it("coalesces a reconnect that arrives immediately after the previous resume", () => {
    // now === lastResumeAt → sinceLastResume === 0, so the full window remains.
    const remaining = shouldDebounceReconnect(5_000, 5_000, WINDOW_MS);
    expect(remaining).toBe(WINDOW_MS);
  });

  it("returns null when exactly on the window boundary", () => {
    // sinceLastResume === windowMs is NOT inside the window — the next
    // reconnect should fire immediately.
    expect(shouldDebounceReconnect(1_000, 2_500, WINDOW_MS)).toBeNull();
  });

  it("returns null when the window has elapsed", () => {
    expect(shouldDebounceReconnect(1_000, 5_000, WINDOW_MS)).toBeNull();
  });

  it("returns a small remaining delay at the far edge of the window", () => {
    // 1ms before the window closes → 1ms left.
    const remaining = shouldDebounceReconnect(1_000, 2_499, WINDOW_MS);
    expect(remaining).toBe(1);
  });

  it("collapses a burst of reconnects into one debounced scheduling", () => {
    // Simulates the browser tab-throttle storm: three reconnect calls fire
    // within a single second after the last resume. Only the first slot
    // would actually run; subsequent calls must always be coalesced.
    const lastResumeAt = 10_000;
    const firstCallRemaining = shouldDebounceReconnect(
      lastResumeAt,
      10_100,
      WINDOW_MS,
    );
    const secondCallRemaining = shouldDebounceReconnect(
      lastResumeAt,
      10_200,
      WINDOW_MS,
    );
    const thirdCallRemaining = shouldDebounceReconnect(
      lastResumeAt,
      10_300,
      WINDOW_MS,
    );
    expect(firstCallRemaining).toBe(1_400);
    expect(secondCallRemaining).toBe(1_300);
    expect(thirdCallRemaining).toBe(1_200);
  });

  it("allows a reconnect to fire immediately once the window has passed", () => {
    // After the window expires, a retry that came in earlier can now fire
    // rather than stalling the loop. Guards against the regression that
    // motivated the coalesce-instead-of-drop fix.
    const lastResumeAt = 10_000;
    expect(
      shouldDebounceReconnect(lastResumeAt, 10_500, WINDOW_MS),
    ).not.toBeNull();
    expect(shouldDebounceReconnect(lastResumeAt, 11_500, WINDOW_MS)).toBeNull();
  });

  it("honours a custom windowMs value", () => {
    // Shouldn't hard-code 1500 anywhere: the helper is generic over the
    // window.
    expect(shouldDebounceReconnect(1_000, 1_500, 2_000)).toBe(1_500);
    expect(shouldDebounceReconnect(1_000, 3_500, 2_000)).toBeNull();
  });
});

describe("resolveSessionDryRun", () => {
  it("returns false when queryData is null", () => {
    expect(resolveSessionDryRun(null)).toBe(false);
  });

  it("returns false when queryData is undefined", () => {
    expect(resolveSessionDryRun(undefined)).toBe(false);
  });

  it("returns false when status is not 200", () => {
    expect(resolveSessionDryRun({ status: 404 })).toBe(false);
  });

  it("returns false when status is 200 but metadata.dry_run is false", () => {
    expect(
      resolveSessionDryRun({
        status: 200,
        data: { metadata: { dry_run: false } },
      }),
    ).toBe(false);
  });

  it("returns false when status is 200 but metadata is missing", () => {
    expect(resolveSessionDryRun({ status: 200, data: {} })).toBe(false);
  });

  it("returns true when status is 200 and metadata.dry_run is true", () => {
    expect(
      resolveSessionDryRun({
        status: 200,
        data: { metadata: { dry_run: true } },
      }),
    ).toBe(true);
  });
});

describe("getCopilotAuthHeaders", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetSystemHeaders.mockReturnValue({});
  });

  it("returns Authorization header when token is present and no impersonation active", async () => {
    mockGetWebSocketToken.mockResolvedValue({
      token: "test-jwt-token",
      error: undefined,
    });

    const headers = await getCopilotAuthHeaders();

    expect(headers).toEqual({ Authorization: "Bearer test-jwt-token" });
  });

  it("names the browser's PostHog session so the chat's events join it", async () => {
    mockGetWebSocketToken.mockResolvedValue({
      token: "test-jwt-token",
      error: undefined,
    });
    vi.mocked(getPostHogSessionHeader).mockReturnValueOnce({
      "X-PostHog-Session-Id": "posthog-session-1",
    });

    const headers = await getCopilotAuthHeaders();

    expect(headers).toEqual({
      Authorization: "Bearer test-jwt-token",
      "X-PostHog-Session-Id": "posthog-session-1",
    });
  });

  it("includes X-Act-As-User-Id header when impersonation is active", async () => {
    mockGetWebSocketToken.mockResolvedValue({
      token: "test-jwt-token",
      error: undefined,
    });
    mockGetSystemHeaders.mockReturnValue({
      [IMPERSONATION_HEADER_NAME]: "impersonated-user-123",
    });

    const headers = await getCopilotAuthHeaders();

    expect(headers).toEqual({
      Authorization: "Bearer test-jwt-token",
      [IMPERSONATION_HEADER_NAME]: "impersonated-user-123",
    });
  });

  it("throws when getWebSocketToken returns an error", async () => {
    mockGetWebSocketToken.mockResolvedValue({
      token: null,
      error: "Token fetch failed",
    });

    await expect(getCopilotAuthHeaders()).rejects.toThrow(
      "Authentication failed — please sign in again.",
    );
  });

  it("throws when getWebSocketToken returns no token and no error", async () => {
    mockGetWebSocketToken.mockResolvedValue({
      token: null,
      error: undefined,
    });

    await expect(getCopilotAuthHeaders()).rejects.toThrow(
      "Authentication failed — please sign in again.",
    );
  });
});

describe("isEngineSwitchPart", () => {
  it("reports a switch for a data-mode-changed part naming either engine", () => {
    expect(
      isEngineSwitchPart({
        type: "data-mode-changed",
        data: { mode: "extended_thinking" },
      }),
    ).toBe(true);
    expect(
      isEngineSwitchPart({
        type: "data-mode-changed",
        data: { mode: "fast" },
      }),
    ).toBe(true);
  });

  it("ignores other data part types", () => {
    expect(
      isEngineSwitchPart({ type: "data-status", data: { mode: "fast" } }),
    ).toBe(false);
  });

  it("ignores unknown or missing engines", () => {
    expect(
      isEngineSwitchPart({
        type: "data-mode-changed",
        data: { mode: "turbo" },
      }),
    ).toBe(false);
    expect(isEngineSwitchPart({ type: "data-mode-changed" })).toBe(false);
    expect(
      isEngineSwitchPart({ type: "data-mode-changed", data: "fast" }),
    ).toBe(false);
  });
});
