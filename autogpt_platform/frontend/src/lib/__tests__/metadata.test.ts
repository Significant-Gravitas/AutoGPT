import { afterEach, describe, expect, test, vi } from "vitest";
import { buildPageMetadata, getSiteUrl } from "../metadata";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("buildPageMetadata", () => {
  test("emits a large-image card when an image is given", () => {
    const metadata = buildPageMetadata({
      title: "An Agent",
      description: "What it does",
      path: "/marketplace/agent/pwuts/an-agent",
      images: ["https://cdn.example.com/agent.png"],
      type: "article",
    });

    expect(metadata.openGraph?.images).toEqual([
      "https://cdn.example.com/agent.png",
    ]);
    expect(metadata.twitter).toMatchObject({ card: "summary_large_image" });
    expect(metadata.twitter?.images).toEqual([
      "https://cdn.example.com/agent.png",
    ]);
    expect(metadata.openGraph).toMatchObject({
      title: "An Agent",
      description: "What it does",
      siteName: "AutoGPT",
      type: "article",
    });
  });

  test.each([
    ["no images key", undefined],
    ["an empty list", []],
    ["a null entry", [null]],
    ["an empty string", [""]],
  ])("omits og:image given %s", (_label, images) => {
    const metadata = buildPageMetadata({ title: "Untitled", images });

    expect(metadata.openGraph).not.toHaveProperty("images");
    expect(metadata.twitter).not.toHaveProperty("images");
    expect(metadata.twitter).toMatchObject({ card: "summary" });
  });

  test("resolves the canonical and og:url against the site URL", () => {
    vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", "https://platform.agpt.co");

    const metadata = buildPageMetadata({
      title: "Marketplace",
      path: "/marketplace",
    });

    expect(metadata.alternates?.canonical).toBe(
      "https://platform.agpt.co/marketplace",
    );
    expect(metadata.openGraph?.url).toBe(
      "https://platform.agpt.co/marketplace",
    );
  });

  test("leaves the canonical unset when no path is given", () => {
    const metadata = buildPageMetadata({ title: "Marketplace" });

    expect(metadata.alternates).toBeUndefined();
    expect(metadata.openGraph?.url).toBeUndefined();
  });

  test("drops an empty description rather than emitting an empty tag", () => {
    const metadata = buildPageMetadata({ title: "A Creator", description: "" });

    expect(metadata.description).toBeUndefined();
    expect(metadata.openGraph?.description).toBeUndefined();
  });
});

describe("getSiteUrl", () => {
  test("prefers the configured frontend base URL", () => {
    vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", "https://platform.agpt.co");
    vi.stubEnv("VERCEL_URL", "some-deployment.vercel.app");

    expect(getSiteUrl()).toBe("https://platform.agpt.co");
  });

  test("falls back to VERCEL_URL as an absolute origin", () => {
    vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", "");
    vi.stubEnv("VERCEL_URL", "some-deployment.vercel.app");

    expect(getSiteUrl()).toBe("https://some-deployment.vercel.app");
  });

  test("falls back to localhost with neither set", () => {
    vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", "");
    vi.stubEnv("VERCEL_URL", "");

    expect(getSiteUrl()).toBe("http://localhost:3000");
  });

  test("skips a configured origin that is not a valid URL", () => {
    vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", "platform.agpt.co");
    vi.stubEnv("VERCEL_URL", "some-deployment.vercel.app");

    expect(getSiteUrl()).toBe("https://some-deployment.vercel.app");
  });

  test("always returns an origin new URL() accepts", () => {
    vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", "not a url");
    vi.stubEnv("VERCEL_URL", "also not a url");

    expect(() => new URL(getSiteUrl())).not.toThrow();
  });
});
