import { describe, expect, test } from "vitest";
import { sanitizeAuthNext } from "./auth-redirect";

describe("sanitizeAuthNext", () => {
  test("returns null when the param is missing or empty", () => {
    expect(sanitizeAuthNext(null)).toBeNull();
    expect(sanitizeAuthNext(undefined)).toBeNull();
    expect(sanitizeAuthNext("")).toBeNull();
  });

  test("rejects absolute URLs and other off-site redirects", () => {
    expect(sanitizeAuthNext("https://phishing.site")).toBeNull();
    expect(sanitizeAuthNext("http://example.com/path")).toBeNull();
    expect(sanitizeAuthNext("javascript:alert(1)")).toBeNull();
    expect(sanitizeAuthNext("mailto:victim@example.com")).toBeNull();
  });

  test("rejects protocol-relative paths so `//evil.com` cannot redirect off-site", () => {
    expect(sanitizeAuthNext("//evil.com")).toBeNull();
    expect(sanitizeAuthNext("//evil.com/foo")).toBeNull();
  });

  test("rejects paths that do not start with /", () => {
    expect(sanitizeAuthNext("library")).toBeNull();
    expect(sanitizeAuthNext("..\\evil")).toBeNull();
  });

  test("accepts same-origin relative paths verbatim", () => {
    expect(sanitizeAuthNext("/library")).toBe("/library");
    expect(sanitizeAuthNext("/onboarding?step=2")).toBe("/onboarding?step=2");
    expect(sanitizeAuthNext("/copilot#section")).toBe("/copilot#section");
    expect(sanitizeAuthNext("/")).toBe("/");
  });
});
