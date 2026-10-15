import { afterEach, describe, expect, it, vi } from "vitest";
import { getPostHogBaseProperties } from "../posthog-base-properties";

afterEach(() => {
  vi.unstubAllEnvs();
});

function stubDeployment(appEnv: string, vercelEnv: string, targetEnv?: string) {
  vi.stubEnv("NEXT_PUBLIC_APP_ENV", appEnv);
  vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", vercelEnv);
  vi.stubEnv("NEXT_PUBLIC_VERCEL_TARGET_ENV", targetEnv);
  vi.stubEnv("VERCEL_TARGET_ENV", undefined);
}

describe("getPostHogBaseProperties", () => {
  it("names the browser as the emitter and carries the app environment", () => {
    stubDeployment("prod", "production", "production");

    expect(getPostHogBaseProperties()).toEqual({
      source: "web",
      environment: "prod",
    });
  });

  it("tells a Vercel preview apart from production", () => {
    stubDeployment("prod", "preview", "preview");

    expect(getPostHogBaseProperties().environment).toBe("preview");
  });

  it("tells a preview apart when the build carries no target environment", () => {
    stubDeployment("prod", "preview");

    expect(getPostHogBaseProperties().environment).toBe("preview");
  });

  it("reports the dev deployment as dev although Vercel calls its staging environment a preview", () => {
    stubDeployment("dev", "preview", "staging");

    expect(getPostHogBaseProperties()).toEqual({
      source: "web",
      environment: "dev",
    });
  });
});
