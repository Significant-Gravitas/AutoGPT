import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  chatSharePath,
  chatShareUrl,
  executionSharePath,
  executionShareUrl,
  sharedChatFilePattern,
  sharedChatFileUrl,
} from "../routes";

describe("share viewer path builders", () => {
  test("chatSharePath returns the canonical viewer route", () => {
    expect(chatSharePath("abc-123")).toBe("/share/chat/abc-123");
  });

  test("executionSharePath returns the canonical viewer route", () => {
    expect(executionSharePath("abc-123")).toBe("/share/abc-123");
  });
});

describe("share viewer absolute URL builders", () => {
  const ORIGINAL_ENV = process.env.NEXT_PUBLIC_FRONTEND_BASE_URL;

  beforeEach(() => {
    delete process.env.NEXT_PUBLIC_FRONTEND_BASE_URL;
  });

  afterEach(() => {
    if (ORIGINAL_ENV === undefined) {
      delete process.env.NEXT_PUBLIC_FRONTEND_BASE_URL;
    } else {
      process.env.NEXT_PUBLIC_FRONTEND_BASE_URL = ORIGINAL_ENV;
    }
    vi.unstubAllGlobals();
  });

  test("chatShareUrl uses window.location.origin when env var is unset", () => {
    vi.stubGlobal("window", {
      location: { origin: "https://app.example.com" },
    });
    expect(chatShareUrl("abc-123")).toBe(
      "https://app.example.com/share/chat/abc-123",
    );
  });

  test("executionShareUrl uses NEXT_PUBLIC_FRONTEND_BASE_URL when set", () => {
    process.env.NEXT_PUBLIC_FRONTEND_BASE_URL = "https://shared.example.com";
    vi.stubGlobal("window", {
      location: { origin: "https://app.example.com" },
    });
    expect(executionShareUrl("abc-123")).toBe(
      "https://shared.example.com/share/abc-123",
    );
  });

  test("chatShareUrl prefers env var over window.location", () => {
    process.env.NEXT_PUBLIC_FRONTEND_BASE_URL = "https://shared.example.com";
    vi.stubGlobal("window", {
      location: { origin: "https://app.example.com" },
    });
    expect(chatShareUrl("abc-123")).toBe(
      "https://shared.example.com/share/chat/abc-123",
    );
  });
});

describe("sharedChatFileUrl", () => {
  test("returns the public allowlist-gated download path", () => {
    const url = sharedChatFileUrl(
      "550e8400-e29b-41d4-a716-446655440000",
      "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    );
    expect(url).toBe(
      "/api/proxy/api/public/shared/chats/550e8400-e29b-41d4-a716-446655440000/files/6ba7b810-9dad-11d1-80b4-00c04fd430c8/download",
    );
  });
});

describe("sharedChatFilePattern", () => {
  const SHARE_TOKEN = "550e8400-e29b-41d4-a716-446655440000";
  const FILE_ID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8";

  test("matches the URL built by sharedChatFileUrl for the same token", () => {
    const pattern = sharedChatFilePattern(SHARE_TOKEN);
    const url = sharedChatFileUrl(SHARE_TOKEN, FILE_ID);
    const match = url.match(pattern);
    expect(match?.[1]).toBe(FILE_ID);
  });

  test("rejects URLs built for a different token (no cross-share match)", () => {
    const pattern = sharedChatFilePattern(SHARE_TOKEN);
    const otherToken = "00000000-0000-0000-0000-000000000000";
    const url = sharedChatFileUrl(otherToken, FILE_ID);
    expect(url.match(pattern)).toBeNull();
  });

  test("rejects non-UUID file IDs", () => {
    const pattern = sharedChatFilePattern(SHARE_TOKEN);
    const badUrl = `/api/proxy/api/public/shared/chats/${SHARE_TOKEN}/files/not-a-uuid/download`;
    expect(badUrl.match(pattern)).toBeNull();
  });

  test("rejects workspace-file URLs (anchored on share prefix)", () => {
    const pattern = sharedChatFilePattern(SHARE_TOKEN);
    const workspaceUrl = `/api/proxy/api/workspace/files/${FILE_ID}/download`;
    expect(workspaceUrl.match(pattern)).toBeNull();
  });

  test("escapes regex metacharacters in token (defensive)", () => {
    // Tokens are UUIDs in production, but the helper must still be
    // safe against a caller passing a token with regex special chars.
    const pattern = sharedChatFilePattern("token.with*special?chars");
    const url = `/api/proxy/api/public/shared/chats/token.with*special?chars/files/${FILE_ID}/download`;
    expect(url.match(pattern)?.[1]).toBe(FILE_ID);
    // And a similar URL with different escaping is rejected.
    const evilUrl = `/api/proxy/api/public/shared/chats/tokenXwithYspecialZchars/files/${FILE_ID}/download`;
    expect(evilUrl.match(pattern)).toBeNull();
  });
});
