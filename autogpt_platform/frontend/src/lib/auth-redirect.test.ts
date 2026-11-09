import { describe, expect, test } from "vitest";
import {
  getAuthDeepLink,
  isDefaultLanding,
  sanitizeAuthNext,
  withAuthTracking,
} from "./auth-redirect";

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
    expect(sanitizeAuthNext("/library#section")).toBe("/library#section");
    expect(sanitizeAuthNext("/")).toBe("/");
  });

  // /copilot is only a server redirect to /home, and a client navigation into it
  // from a (no-navbar) page crashes the App Router with React #310.
  test.each([
    ["/copilot", "/home"],
    ["/copilot/", "/home"],
    ["/copilot?sessionId=abc", "/home?sessionId=abc"],
    ["/copilot#frag", "/home#frag"],
    ["/copilot/?sessionId=abc#frag", "/home?sessionId=abc#frag"],
  ])("maps %s to %s so the redirect hop is skipped", (value, expected) => {
    expect(sanitizeAuthNext(value)).toBe(expected);
  });

  test.each([
    "/copilots",
    "/copilot/styleguide",
    "/home",
    "/home?sessionId=abc",
  ])("leaves %s untouched", (value) => {
    expect(sanitizeAuthNext(value)).toBe(value);
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

// `/login?next=/home` is where every signed-out visit to the app root lands.
// Such a `next` names no page the user chose, so it must not send someone who
// hasn't finished onboarding to Home.
describe("isDefaultLanding", () => {
  test.each([
    "/",
    "/home",
    "/home/",
    "/home?",
    "/home#",
    "/?ref=newsletter",
    "/#top",
    "/home?utm_source=newsletter&utm_campaign=launch",
    "/home?gclid=abc",
    "/home?fbclid=abc&ref=x",
  ])("%s is a default landing", (value) => {
    expect(isDefaultLanding(value)).toBe(true);
  });

  test.each([
    "/home?sessionId=abc",
    "/home?sessionId=abc&utm_source=newsletter",
    "/home?autosubmit=true",
    "/home#chat",
    "/homework",
    "/home/settings",
    "/library",
    "/library?utm_source=newsletter",
    "/marketplace/agent/autogpt/some-workflow",
    "/onboarding",
  ])("%s is not", (value) => {
    expect(isDefaultLanding(value)).toBe(false);
  });
});

describe("getAuthDeepLink", () => {
  test.each([
    "/",
    "/home",
    "/home/",
    "/copilot",
    "/copilot/",
    "/copilot?utm_source=newsletter",
    "/home?utm_campaign=launch",
  ])("drops the default landing %s so onboarding decides", (value) => {
    expect(getAuthDeepLink(value)).toBeNull();
  });

  test("drops what sanitizeAuthNext drops", () => {
    expect(getAuthDeepLink(null)).toBeNull();
    expect(getAuthDeepLink("")).toBeNull();
    expect(getAuthDeepLink("https://phishing.site")).toBeNull();
    expect(getAuthDeepLink("//evil.com/home")).toBeNull();
    expect(getAuthDeepLink("/\\evil.com")).toBeNull();
  });

  test.each([
    ["/library", "/library"],
    ["/marketplace/agent/autogpt/x", "/marketplace/agent/autogpt/x"],
    ["/home?sessionId=abc", "/home?sessionId=abc"],
    ["/copilot?sessionId=abc", "/home?sessionId=abc"],
    [
      "/auth/integrations/oauth_callback?code=1",
      "/auth/integrations/oauth_callback?code=1",
    ],
  ])("keeps the real deep link %s", (value, expected) => {
    expect(getAuthDeepLink(value)).toBe(expected);
  });
});

// A campaign or ad visit lands on /login?next=/home?utm_source=…, so the
// page an auth flow sends the user to must carry those params to analytics.
describe("withAuthTracking", () => {
  test.each([
    [
      "/onboarding",
      "/home?utm_source=newsletter",
      "/onboarding?utm_source=newsletter",
    ],
    [
      "/onboarding",
      "/copilot?utm_source=ads&utm_campaign=launch&gclid=abc",
      "/onboarding?utm_source=ads&utm_campaign=launch&gclid=abc",
    ],
    ["/copilot", "/home?_gl=1*abc", "/copilot?_gl=1*abc"],
    ["/onboarding", "/?utm_source=x&step=2", "/onboarding?utm_source=x"],
  ])(
    "adds the tracking params to %s for next=%s",
    (destination, next, expected) => {
      expect(withAuthTracking(destination, next)).toBe(expected);
    },
  );

  test.each([
    ["/onboarding", null],
    ["/onboarding", "/home"],
    ["/onboarding", "/library?utm_source=x"],
    ["/onboarding", "https://evil.example/home?utm_source=x"],
  ])("leaves %s alone for next=%s", (destination, next) => {
    expect(withAuthTracking(destination, next)).toBe(destination);
  });
});
