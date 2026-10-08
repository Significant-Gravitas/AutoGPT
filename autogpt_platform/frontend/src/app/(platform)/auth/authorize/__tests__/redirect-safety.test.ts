import { describe, expect, it } from "vitest";

import {
  buildOAuthAccessDeniedRedirect,
  isRegisteredRedirectUri,
  isSafeOAuthRedirectUrl,
} from "../redirect-safety";

const GOOD = "https://example.com/callback";
const EVIL = "https://evil.example/phish";

describe("isRegisteredRedirectUri", () => {
  it("accepts an exact registered URI", () => {
    expect(
      isRegisteredRedirectUri(GOOD, [GOOD, "http://localhost:3000/cb"]),
    ).toBe(true);
  });

  it("rejects an evil URI not in the allowlist", () => {
    expect(isRegisteredRedirectUri(EVIL, [GOOD])).toBe(false);
  });

  it("rejects when registered list is missing", () => {
    expect(isRegisteredRedirectUri(EVIL, null)).toBe(false);
    expect(isRegisteredRedirectUri(EVIL, undefined)).toBe(false);
    expect(isRegisteredRedirectUri(EVIL, [])).toBe(false);
  });

  it("rejects null/empty redirect", () => {
    expect(isRegisteredRedirectUri(null, [GOOD])).toBe(false);
    expect(isRegisteredRedirectUri("", [GOOD])).toBe(false);
  });

  it("matches exactly, not by prefix", () => {
    expect(
      isRegisteredRedirectUri("http://localhost:3000.evil.example/cb", [
        "http://localhost:3000",
      ]),
    ).toBe(false);
  });
});

describe("isSafeOAuthRedirectUrl", () => {
  it("accepts backend success redirect to registered URI", () => {
    expect(isSafeOAuthRedirectUrl(`${GOOD}?code=abc&state=s`, GOOD)).toBe(true);
  });

  it("accepts backend error redirect to registered URI", () => {
    expect(
      isSafeOAuthRedirectUrl(`${GOOD}?error=invalid_scope&state=s`, GOOD),
    ).toBe(true);
  });

  it("rejects evil redirect_url even with query params", () => {
    expect(
      isSafeOAuthRedirectUrl(`${EVIL}?error=invalid_client&state=s`, GOOD),
    ).toBe(false);
  });

  it("rejects path-traversal style mismatches", () => {
    expect(
      isSafeOAuthRedirectUrl("https://example.com/callback/../evil", GOOD),
    ).toBe(false);
  });

  it("rejects when registered URI is missing", () => {
    expect(isSafeOAuthRedirectUrl(`${GOOD}?code=x`, null)).toBe(false);
  });

  it("rejects opaque / unparseable URLs", () => {
    expect(isSafeOAuthRedirectUrl("not a url", GOOD)).toBe(false);
  });

  it("rejects a different custom-scheme callback", () => {
    expect(
      isSafeOAuthRedirectUrl("evil://other?code=x", "myapp://callback"),
    ).toBe(false);
    expect(
      isSafeOAuthRedirectUrl("myapp://callback?code=x", "myapp://callback"),
    ).toBe(true);
  });
});

describe("buildOAuthAccessDeniedRedirect", () => {
  it("builds access_denied query on the registered URI", () => {
    const url = buildOAuthAccessDeniedRedirect(GOOD, "csrf");
    expect(url.startsWith(`${GOOD}?`)).toBe(true);
    expect(url).toContain("error=access_denied");
    expect(url).toContain("state=csrf");
    expect(url).not.toContain(EVIL);
  });

  it("preserves an existing query string on the registered URI", () => {
    const url = new URL(
      buildOAuthAccessDeniedRedirect(`${GOOD}?tenant=a`, "csrf"),
    );
    expect(`${url.origin}${url.pathname}`).toBe(GOOD);
    expect(url.searchParams.get("tenant")).toBe("a");
    expect(url.searchParams.get("error")).toBe("access_denied");
    expect(url.searchParams.get("state")).toBe("csrf");
  });
});
