import { join } from "node:path";
import type { NextConfig } from "next";
import { hasLocalMatch } from "next/dist/shared/lib/match-local-pattern";
import { describe, expect, it, vi } from "vitest";

async function loadConfig() {
  const { default: nextConfig } = await vi.importActual<{
    default: NextConfig;
  }>(join(process.cwd(), "next.config.mjs"));
  return nextConfig;
}

describe("next.config.mjs image optimizer", () => {
  it("caps the pixels it decodes well below Next.js's default", async () => {
    const { experimental } = await loadConfig();

    // sharp reads 0 as "no limit".
    expect(experimental?.imgOptMaxInputPixels).toBeGreaterThan(0);
    expect(experimental?.imgOptMaxInputPixels).toBeLessThanOrEqual(50_000_000);
  });

  it.each([
    "/api/store/media/user-123/images/thumbnail.png",
    "/API/store/media/user-123/images/thumbnail.png",
    "/api/proxy/api/store/media/user-123/images/thumbnail.png",
    "/api/proxy/api/workspace/files/file-123/download",
    "/images/../api/store/media/user-123/images/thumbnail.png",
  ])("does not optimize %s", async (path) => {
    const { images } = await loadConfig();

    expect(hasLocalMatch(images?.localPatterns, path)).toBe(false);
  });

  it.each([
    "/placeholder.png",
    "/autogpt-characters/v1.1/otto/neutral/128.webp",
    "/experts/covers/otto-cover-3.png",
    "/images/tour-og.png",
    "/integrations/github.png",
    "/storybook/atoms.png",
  ])("optimizes the public image %s", async (path) => {
    const { images } = await loadConfig();

    expect(hasLocalMatch(images?.localPatterns, path)).toBe(true);
  });
});
