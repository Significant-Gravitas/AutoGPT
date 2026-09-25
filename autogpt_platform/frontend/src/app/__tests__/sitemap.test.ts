import type { ExpertTemplate } from "@/app/api/__generated__/models/expertTemplate";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { listExpertTemplates } from "@/app/api/__generated__/endpoints/experts/experts";
import sitemap from "../sitemap";

vi.mock("@/app/api/__generated__/endpoints/experts/experts", () => ({
  listExpertTemplates: vi.fn(),
}));

const SITE_URL = "https://platform.agpt.co";

function template(overrides: Partial<ExpertTemplate>): ExpertTemplate {
  return {
    id: "expert-id",
    name: "Expert",
    avatar_url: null,
    role: "analyst",
    tagline: null,
    bio: null,
    skills: [],
    identity: "",
    voice_preferences: "",
    boundaries: "",
    protected_soul_rules: [],
    is_template: true,
    source_template_id: null,
    is_archived: false,
    workflows: [],
    ...overrides,
  };
}

function mockTemplates(templates: ExpertTemplate[]) {
  vi.mocked(listExpertTemplates).mockResolvedValue({
    data: templates,
    status: 200,
    headers: new Headers(),
  });
}

function urlsOf(entries: Awaited<ReturnType<typeof sitemap>>): string[] {
  return entries.map((entry) => entry.url);
}

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", SITE_URL);
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("sitemap", () => {
  test("lists the marketplace and one absolute URL per public expert", async () => {
    mockTemplates([
      template({ id: "d91d9897-5c65-45c6-ba16-0dd5c24404ac" }),
      template({ id: "7a25f32e-26e4-4a4e-9902-aed163e61c1d" }),
    ]);

    const urls = urlsOf(await sitemap());

    expect(urls).toEqual([
      `${SITE_URL}/marketplace`,
      `${SITE_URL}/marketplace/experts/d91d9897-5c65-45c6-ba16-0dd5c24404ac`,
      `${SITE_URL}/marketplace/experts/7a25f32e-26e4-4a4e-9902-aed163e61c1d`,
    ]);
  });

  test("builds URLs from getSiteUrl so dev and prod differ", async () => {
    vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", "https://dev-builder.agpt.co");
    mockTemplates([template({ id: "abc" })]);

    const urls = urlsOf(await sitemap());

    expect(urls).toEqual([
      "https://dev-builder.agpt.co/marketplace",
      "https://dev-builder.agpt.co/marketplace/experts/abc",
    ]);
    expect(urls.every((url) => new URL(url).protocol === "https:")).toBe(true);
  });

  test("excludes archived templates", async () => {
    mockTemplates([
      template({ id: "live" }),
      template({ id: "gone", is_archived: true }),
    ]);

    const urls = urlsOf(await sitemap());

    expect(urls).toContain(`${SITE_URL}/marketplace/experts/live`);
    expect(urls).not.toContain(`${SITE_URL}/marketplace/experts/gone`);
  });

  test("excludes rows that are not templates", async () => {
    mockTemplates([
      template({ id: "live" }),
      template({ id: "hired", is_template: false }),
    ]);

    const urls = urlsOf(await sitemap());

    expect(urls).toContain(`${SITE_URL}/marketplace/experts/live`);
    expect(urls).not.toContain(`${SITE_URL}/marketplace/experts/hired`);
  });

  test("still returns the static entries when the fetch throws", async () => {
    vi.mocked(listExpertTemplates).mockRejectedValue(new Error("backend down"));

    const urls = urlsOf(await sitemap());

    expect(urls).toEqual([`${SITE_URL}/marketplace`]);
    expect(console.error).toHaveBeenCalled();
  });

  test("still returns the static entries on a non-200 response", async () => {
    vi.mocked(listExpertTemplates).mockResolvedValue({
      data: { detail: [] },
      status: 422,
      headers: new Headers(),
    });

    const urls = urlsOf(await sitemap());

    expect(urls).toEqual([`${SITE_URL}/marketplace`]);
  });

  test("never lists private pages", async () => {
    mockTemplates([template({ id: "live" })]);

    const urls = urlsOf(await sitemap());
    const privatePrefixes = [
      "/copilot",
      "/library",
      "/settings",
      "/team",
      "/build",
      "/home",
      "/admin",
      "/profile",
    ];

    for (const url of urls) {
      const path = new URL(url).pathname;
      expect(privatePrefixes.some((p) => path.startsWith(p))).toBe(false);
    }
  });
});
