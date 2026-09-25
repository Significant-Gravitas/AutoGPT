import { ADMIN_PAGES, PROTECTED_PAGES } from "@/lib/auth/helpers";
import { afterEach, describe, expect, test, vi } from "vitest";
import robots from "../robots";

afterEach(() => {
  vi.unstubAllEnvs();
});

function getRules() {
  const { rules } = robots();
  return Array.isArray(rules) ? rules[0] : rules;
}

describe("robots", () => {
  test("allows everything by default", () => {
    const rules = getRules();

    expect(rules.userAgent).toBe("*");
    expect(rules.allow).toBe("/");
  });

  test("disallows the protected, admin, api and auth prefixes", () => {
    const disallow = getRules().disallow;

    for (const prefix of [
      ...PROTECTED_PAGES,
      ...ADMIN_PAGES,
      "/api",
      "/auth",
    ]) {
      expect(disallow).toContain(prefix);
    }
  });

  test("leaves the marketplace crawlable", () => {
    const disallow = getRules().disallow;
    const list = Array.isArray(disallow) ? disallow : [disallow];

    expect(list.some((p) => p?.startsWith("/marketplace"))).toBe(false);
  });

  test("points at the sitemap on the site URL", () => {
    vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", "https://platform.agpt.co");

    expect(robots().sitemap).toBe("https://platform.agpt.co/sitemap.xml");
  });
});
