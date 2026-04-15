import { beforeEach, describe, expect, it, vi } from "vitest";
import { IMPERSONATION_HEADER_NAME } from "@/lib/constants";
import {
  getCopilotAuthHeaders,
  getSendSuppressionReason,
  resolveSessionDryRun,
} from "../helpers";
import type { UIMessage } from "ai";

vi.mock("@/lib/supabase/actions", () => ({
  getWebSocketToken: vi.fn(),
}));

vi.mock("@/lib/impersonation", () => ({
  getSystemHeaders: vi.fn(),
}));

import { getWebSocketToken } from "@/lib/supabase/actions";
import { getSystemHeaders } from "@/lib/impersonation";

const mockGetWebSocketToken = vi.mocked(getWebSocketToken);
const mockGetSystemHeaders = vi.mocked(getSystemHeaders);

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

// ─── getSendSuppressionReason ─────────────────────────────────────────────────

function makeUserMsg(text: string): UIMessage {
  return {
    id: "msg-1",
    role: "user",
    content: text,
    parts: [{ type: "text", text }],
  } as UIMessage;
}

describe("getSendSuppressionReason", () => {
  it("returns null when no dedup context exists (fresh ref)", () => {
    const result = getSendSuppressionReason({
      text: "hello",
      isReconnectScheduled: false,
      lastSubmittedText: null,
      messages: [],
    });
    expect(result).toBeNull();
  });

  it("returns 'reconnecting' when reconnect is scheduled regardless of text", () => {
    const result = getSendSuppressionReason({
      text: "hello",
      isReconnectScheduled: true,
      lastSubmittedText: null,
      messages: [],
    });
    expect(result).toBe("reconnecting");
  });

  it("returns 'duplicate' when same text was submitted and is the last user message", () => {
    // This is the core regression test: after a successful turn the ref
    // is intentionally NOT cleared to null, so submitting the same text
    // again is caught here.
    const result = getSendSuppressionReason({
      text: "hello",
      isReconnectScheduled: false,
      lastSubmittedText: "hello",
      messages: [makeUserMsg("hello")],
    });
    expect(result).toBe("duplicate");
  });

  it("returns null when same ref text but different last user message (different question)", () => {
    // User asked "hello" before, got a reply, then asked a different question
    // — the last user message in chat is now different, so no suppression.
    const result = getSendSuppressionReason({
      text: "hello",
      isReconnectScheduled: false,
      lastSubmittedText: "hello",
      messages: [makeUserMsg("hello"), makeUserMsg("something else")],
    });
    expect(result).toBeNull();
  });

  it("returns null when text differs from lastSubmittedText", () => {
    const result = getSendSuppressionReason({
      text: "new question",
      isReconnectScheduled: false,
      lastSubmittedText: "old question",
      messages: [makeUserMsg("old question")],
    });
    expect(result).toBeNull();
  });
});
