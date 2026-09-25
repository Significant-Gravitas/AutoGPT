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

function toList(value: string | string[] | undefined): string[] {
  if (value === undefined) return [];
  return Array.isArray(value) ? value : [value];
}

// Google's robots.txt matching: the longest matching rule wins, and an allow
// beats a disallow of the same length. Prefixes match the path plus query.
function isCrawlable(url: string): boolean {
  const rules = getRules();
  const longest = (prefixes: string[]) =>
    Math.max(
      -1,
      ...prefixes.filter((p) => url.startsWith(p)).map((p) => p.length),
    );
  return longest(toList(rules.allow)) >= longest(toList(rules.disallow));
}

describe("robots", () => {
  test("allows everything by default", () => {
    const rules = getRules();

    expect(rules.userAgent).toBe("*");
    expect(toList(rules.allow)).toContain("/");
    expect(isCrawlable("/marketplace/experts/expert-1")).toBe(true);
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

  test.each([
    "/api/proxy/api/experts/templates",
    "/api/proxy/api/experts/templates?include_archived=false",
    "/api/proxy/api/integrations/providers/system",
    "/api/proxy/api/store/agents?search_query=email&sorted_by=runs",
    "/api/proxy/api/store/agents/creator/agent-slug",
    "/api/proxy/api/store/creators?featured=true",
    "/api/proxy/api/store/creators/creator",
    "/api/proxy/api/store/categories",
    "/api/proxy/api/store/search?search_query=email",
    "/api/proxy/api/store/skills?page=1&page_size=20",
    "/api/proxy/api/store/skills/my-skill",
    "/api/proxy/api/store/skills/my-skill/files/SKILL.md",
    "/api/proxy/api/store/media/user-1/images/thumb.png",
    "/api/store/media/user-1/images/thumb.png",
  ])("lets crawlers render the marketplace through %s", (url) => {
    expect(isCrawlable(url)).toBe(true);
  });

  test.each([
    "/api/auth/user",
    "/api/proxy/api/auth/user",
    "/api/proxy/api/experts",
    "/api/proxy/api/experts/expert-1",
    "/api/proxy/api/experts/expert-1/credentials",
    "/api/proxy/api/library/agents",
    "/api/proxy/api/store/profile",
    "/api/proxy/api/store/submissions",
    "/api/proxy/api/store/my-unpublished-agents",
    "/api/proxy/api/store/skills/submissions",
    "/api/proxy/api/store/skills/submissions/version-1",
    "/api/proxy/api/integrations/credentials",
    "/auth/login",
  ])("keeps %s off limits", (url) => {
    expect(isCrawlable(url)).toBe(false);
  });

  test("points at the sitemap on the site URL", () => {
    vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", "https://platform.agpt.co");

    expect(robots().sitemap).toBe("https://platform.agpt.co/sitemap.xml");
  });
});
