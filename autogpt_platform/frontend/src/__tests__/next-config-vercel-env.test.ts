import { join } from "node:path";
import type { NextConfig } from "next";
import { afterEach, describe, expect, it, vi } from "vitest";

async function loadMirroredEnv() {
  vi.resetModules();
  const { default: nextConfig } = await vi.importActual<{
    default: NextConfig;
  }>(join(process.cwd(), "next.config.mjs"));
  return nextConfig.env;
}

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("next.config.mjs Vercel environment mirror", () => {
  it("mirrors the deployment's environment and target environment into the browser bundle", async () => {
    vi.stubEnv("VERCEL_ENV", "preview");
    vi.stubEnv("VERCEL_TARGET_ENV", "staging");

    expect(await loadMirroredEnv()).toMatchObject({
      NEXT_PUBLIC_VERCEL_ENV: "preview",
      NEXT_PUBLIC_VERCEL_TARGET_ENV: "staging",
    });
  });

  it("mirrors empty strings for a build outside Vercel", async () => {
    vi.stubEnv("VERCEL_ENV", undefined);
    vi.stubEnv("VERCEL_TARGET_ENV", undefined);

    expect(await loadMirroredEnv()).toMatchObject({
      NEXT_PUBLIC_VERCEL_ENV: "",
      NEXT_PUBLIC_VERCEL_TARGET_ENV: "",
    });
  });
});
