import { describe, expect, test } from "vitest";

import { LinkType } from "@/app/api/__generated__/models/linkType";
import {
  getLoginRedirect,
  getPlatformDisplayName,
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
    expect(getLoginRedirect("token-123")).toBe(
      "/login?next=%2Flink%2Ftoken-123",
    );
    expect(getLoginRedirect(null)).toBe("/login?next=%2F");
  });

  test("identifies user link token types", () => {
    expect(isUserLink(LinkType.USER)).toBe(true);
    expect(isUserLink(LinkType.SERVER)).toBe(false);
    expect(isUserLink(undefined)).toBe(false);
  });
});
