import { afterEach, describe, expect, it, vi } from "vitest";
import { getPostHogBaseProperties } from "../posthog-base-properties";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("getPostHogBaseProperties", () => {
  it("names the browser as the emitter and carries the app environment", () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "prod");
    vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", "production");

    expect(getPostHogBaseProperties()).toEqual({
      source: "web",
      environment: "prod",
    });
  });

  it("tells a Vercel preview apart from production", () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "prod");
    vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", "preview");

    expect(getPostHogBaseProperties().environment).toBe("preview");
  });
});
