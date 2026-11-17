import { describe, expect, test } from "vitest";

import { LinkType } from "@/app/api/__generated__/models/linkType";
import {
  getLoginRedirect,
  getPlatformDisplayName,
  getSignupRedirect,
  isUserLink,
  TOKEN_PATTERN,
} from "../helpers";

describe("platform linking helpers", () => {
  test("accepts URL-safe token values up to the backend limit", () => {
    expect(TOKEN_PATTERN.test("abc_123-XYZ")).toBe(true);
    expect(TOKEN_PATTERN.test("a".repeat(64))).toBe(true);
  });

  test("rejects malformed token values", () => {
    expect(TOKEN_PATTERN.test("")).toBe(false);
    expect(TOKEN_PATTERN.test("a".repeat(65))).toBe(false);
    expect(TOKEN_PATTERN.test("abc.123")).toBe(false);
    expect(TOKEN_PATTERN.test("abc/123")).toBe(false);
  });

  test("formats known platforms and falls back for unknown values", () => {
    expect(getPlatformDisplayName("discord")).toBe("Discord");
    expect(getPlatformDisplayName("GITHUB")).toBe("GitHub");
    expect(getPlatformDisplayName("matrix")).toBe("matrix");
    expect(getPlatformDisplayName(null)).toBe("chat platform");
  });

  test("builds the login redirect for valid and missing tokens", () => {
    expect(getLoginRedirect({ token: "token-123" })).toBe(
      "/login?next=%2Flink%2Ftoken-123",
    );
    expect(getLoginRedirect({ token: null })).toBe("/login?next=%2F");
  });

  test("keeps a Telegram sign-in through login and sign-up", () => {
    const telegramAuth = {
      id: "424242",
      auth_date: "1750000000",
      hash: "abc123",
    };
    const next = encodeURIComponent(
      "/link/token-123?id=424242&auth_date=1750000000&hash=abc123",
    );

    expect(getLoginRedirect({ token: "token-123", telegramAuth })).toBe(
      `/login?next=${next}`,
    );
    expect(getSignupRedirect({ token: "token-123", telegramAuth })).toBe(
      `/signup?next=${next}`,
    );
  });

  test("identifies user link token types", () => {
    expect(isUserLink(LinkType.USER)).toBe(true);
    expect(isUserLink(LinkType.SERVER)).toBe(false);
    expect(isUserLink(undefined)).toBe(false);
  });
});
