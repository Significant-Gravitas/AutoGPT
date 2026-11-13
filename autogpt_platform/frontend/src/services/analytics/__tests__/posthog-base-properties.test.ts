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

  it("carries the build's revision", () => {
    stubDeployment("prod", "production", "production");
    vi.stubEnv("NEXT_PUBLIC_REVISION", "0123abcd");

    expect(getPostHogBaseProperties()).toEqual({
      source: "web",
      environment: "prod",
      revision: "0123abcd",
    });
  });

  it("leaves the revision out of a build that has none", () => {
    stubDeployment("prod", "production", "production");
    vi.stubEnv("NEXT_PUBLIC_REVISION", "");

    expect(getPostHogBaseProperties()).not.toHaveProperty("revision");
  });

  it("reports the dev deployment as dev although Vercel calls its staging environment a preview", () => {
    stubDeployment("dev", "preview", "staging");

    expect(getPostHogBaseProperties()).toEqual({
      source: "web",
      environment: "dev",
    });
  });
});
