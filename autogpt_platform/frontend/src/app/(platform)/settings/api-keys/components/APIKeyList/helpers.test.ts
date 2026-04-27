import { describe, expect, test } from "vitest";

import { formatLastUsed, maskAPIKey } from "./helpers";

describe("maskAPIKey", () => {
  test("joins head, a fixed mask, and tail", () => {
    expect(maskAPIKey("pk_live_", "abcd1234")).toBe("pk_live_••••••••abcd1234");
  });

  test("handles empty head and tail", () => {
    expect(maskAPIKey("", "")).toBe("••••••••");
  });
});

describe("formatLastUsed", () => {
  test("returns 'Never used' when last used is null", () => {
    expect(formatLastUsed(null)).toBe("Never used");
  });

  test("returns 'Never used' when last used is undefined", () => {
    expect(formatLastUsed(undefined)).toBe("Never used");
  });

  test("returns a relative-time phrase when last used is a Date", () => {
    const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60 * 1000);
    const result = formatLastUsed(twoHoursAgo);
    expect(result.startsWith("Used ")).toBe(true);
    expect(result.includes("ago")).toBe(true);
  });
});
