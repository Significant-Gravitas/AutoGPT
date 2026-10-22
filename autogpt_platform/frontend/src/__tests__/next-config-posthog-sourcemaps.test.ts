import { join } from "node:path";
import type { NextConfig } from "next";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const uploadKey = "private-upload-key-value";
const publicProjectKey = "public-project-key-value";

type NextConfigFunction = (
  phase: string,
  context: { defaultConfig: NextConfig },
) => Promise<NextConfig>;

async function loadConfig() {
  vi.resetModules();
  const { default: config } = await vi.importActual<{
    default: NextConfig | NextConfigFunction;
  }>(join(process.cwd(), "next.config.mjs"));
  return config;
}

beforeEach(() => {
  vi.stubEnv("NODE_ENV", "production");
  vi.stubEnv("NEXT_PUBLIC_SOURCEMAPS", "true");
  vi.stubEnv("SENTRY_AUTH_TOKEN", "");
  vi.stubEnv("POSTHOG_SOURCEMAP_UPLOAD_KEY", "");
  vi.stubEnv("POSTHOG_SOURCEMAP_PROJECT_ID", "");
  vi.stubEnv("POSTHOG_API_KEY", "");
  vi.stubEnv("POSTHOG_PROJECT_ID", "");
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("next.config.mjs PostHog source-map upload", () => {
  it("uploads with both build-only credentials and keeps the key out of the browser bundle", async () => {
    vi.stubEnv("POSTHOG_SOURCEMAP_UPLOAD_KEY", uploadKey);
    vi.stubEnv("POSTHOG_SOURCEMAP_PROJECT_ID", "117338");

    const config = await loadConfig();

    expect(config).toBeTypeOf("function");
    const resolved = await (config as NextConfigFunction)(
      "phase-production-build",
      { defaultConfig: {} },
    );
    expect(resolved.productionBrowserSourceMaps).toBe(true);
    expect(JSON.stringify(resolved.env)).not.toContain(uploadKey);
    expect(Object.keys(resolved.env ?? {})).not.toContain(
      "POSTHOG_SOURCEMAP_UPLOAD_KEY",
    );
  });

  it.each([
    ["only the upload key", { POSTHOG_SOURCEMAP_UPLOAD_KEY: uploadKey }],
    ["only the project", { POSTHOG_SOURCEMAP_PROJECT_ID: "117338" }],
    // The backend's public project key must never be taken for the upload key.
    [
      "the backend's variable names",
      { POSTHOG_API_KEY: publicProjectKey, POSTHOG_PROJECT_ID: "117338" },
    ],
  ])("skips the upload with %s", async (_, env) => {
    for (const [name, value] of Object.entries(env)) vi.stubEnv(name, value);

    const config = await loadConfig();

    expect(config).toBeTypeOf("object");
    expect(JSON.stringify((config as NextConfig).env)).not.toContain(uploadKey);
  });
});
