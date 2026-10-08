import type { UIMessage } from "ai";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { IMPERSONATION_HEADER_NAME } from "@/lib/constants";
import {
  COPILOT_COMPLETION_NOTIFICATION,
  ORIGINAL_TITLE,
  deduplicateMessages,
  extractSendMessageText,
  formatNotificationTitle,
  getCopilotAuthHeaders,
  getSendSuppressionReason,
  parseSessionIDs,
  isEngineSwitchPart,
  resolveSessionDryRun,
  shouldSuppressDuplicateSend,
} from "../helpers";

vi.mock("@/lib/auth/actions", () => ({
  getWebSocketToken: vi.fn(),
}));

vi.mock("@/lib/impersonation", () => ({
  getSystemHeaders: vi.fn(),
}));

import { getWebSocketToken } from "@/lib/auth/actions";
import { getSystemHeaders } from "@/lib/impersonation";

const mockGetWebSocketToken = vi.mocked(getWebSocketToken);
const mockGetSystemHeaders = vi.mocked(getSystemHeaders);

describe("formatNotificationTitle", () => {
  it("returns base title when count is 0", () => {
    expect(formatNotificationTitle(0)).toBe(ORIGINAL_TITLE);
  });

  it("returns formatted title with count", () => {
    expect(formatNotificationTitle(3)).toBe(
      `(3) New activity - ${ORIGINAL_TITLE}`,
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
      `(1) New activity - ${ORIGINAL_TITLE}`,
    );
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
  it("removes messages with duplicate IDs, keeping the first", () => {
    const msgs = [
      makeMsgWithId("1", "user", "hello"),
      makeMsgWithId("1", "user", "hello again"),
    ];
    expect(deduplicateMessages(msgs)).toEqual([msgs[0]]);
  });

  it("keeps identical content under different ids", () => {
    const msgs = [
      makeMsgWithId("u1", "user", "hello"),
      makeMsgWithId("a1", "assistant", "hi there"),
      makeMsgWithId("a2", "assistant", "hi there"),
    ];
    expect(deduplicateMessages(msgs)).toEqual(msgs);
  });

  it("handles empty message list", () => {
    expect(deduplicateMessages([])).toEqual([]);
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
