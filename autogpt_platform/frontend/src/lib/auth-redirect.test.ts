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

  // Everything below resolves off-origin through the WHATWG URL parser, which
  // is what window.location.href and NextResponse.redirect both use. Each case
  // passes a naive startsWith("/") && !startsWith("//") check, so they are the
  // family of bypasses that guard cannot see.
  describe("off-origin escapes that survive a prefix-only check", () => {
    const escapes = [
      ["backslash host", "/\\evil.com"],
      ["backslash host with path", "/\\evil.com/path"],
      ["double backslash", "\\\\evil.com"],
      ["mixed slash/backslash", "/\\/evil.com"],
      ["backslash after slash", "/\\\\evil.com"],
      ["tab before host", "/\t/evil.com"],
      ["newline before host", "/\n/evil.com"],
      ["carriage return before host", "/\r/evil.com"],
      ["backslash anywhere in path", "/foo\\bar"],
    ] as const;

    test.each(escapes)("rejects %s", (_label, value) => {
      expect(sanitizeAuthNext(value)).toBeNull();
    });

    test("the rejected values really do resolve off-origin", () => {
      // Guards the premise: if a future URL-parser change made these safe, this
      // assertion fails and tells us the guard can be relaxed, rather than the
      // tests silently over-protecting.
      const origin = "https://app.example";
      const offOrigin = escapes
        .map(([, value]) => value)
        .filter((value) => {
          try {
            return new URL(value, origin).origin !== origin;
          } catch {
            return false;
          }
        });
      expect(offOrigin.length).toBeGreaterThan(0);
    });
  });

  test("percent-encoded backslash is covered (searchParams decodes before we see it)", () => {
    const decoded = new URLSearchParams("next=%2F%5Cevil.com").get("next");
    expect(decoded).toBe("/\\evil.com");
    expect(sanitizeAuthNext(decoded)).toBeNull();
  });

  test("keeps accepting ordinary paths that merely contain unusual characters", () => {
    expect(sanitizeAuthNext("/build?flowID=a-b_c.d~e")).toBe(
      "/build?flowID=a-b_c.d~e",
    );
    expect(sanitizeAuthNext("/search?q=a%20b&x=1")).toBe("/search?q=a%20b&x=1");
    expect(sanitizeAuthNext("/path/with-dash/and_underscore")).toBe(
      "/path/with-dash/and_underscore",
    );
  });
});
