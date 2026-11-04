import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Flag, useGetFlag } from "../use-get-flag";

// The flag service has not answered for any key — the exact window
// (load lag, outage, missing flag key) where useGetFlag falls through
// to defaultFlags in production.
vi.mock("../flag-source", () => ({
  useFlagSource: () => ({ value: undefined, resolved: false }),
}));

vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));

const DREAM_FLAGS = [
  Flag.DREAM_PASS_ENABLED,
  Flag.DREAM_PASS_INVALIDATE_ENTITY,
  Flag.DREAM_SKILL_LEARNING_ENABLED,
] as const;

describe("experimental dream flag defaults fail closed", () => {
  beforeEach(() => {
    Object.keys(process.env)
      .filter((k) => k.startsWith("NEXT_PUBLIC_FORCE_FLAG_"))
      .forEach((k) => delete process.env[k]);
  });

  it.each(DREAM_FLAGS)(
    "resolves %s to false when the flag service has not answered, mirroring the backend's default=False gating",
    (flag) => {
      const { result } = renderHook(() => useGetFlag(flag));
      expect(result.current).toBe(false);
    },
  );

  it("still lets local dev force-enable the stack via the env override", () => {
    process.env.NEXT_PUBLIC_FORCE_FLAG_DREAM_PASS_ENABLED = "true";
    const { result } = renderHook(() => useGetFlag(Flag.DREAM_PASS_ENABLED));
    expect(result.current).toBe(true);
  });
});

// If the flag service does not answer — outage, load lag, or a key mismatch —
// this is the only path, and it must resolve false or merging ships the
// half-built org management UI to everyone.
describe("org settings flag default fails closed", () => {
  beforeEach(() => {
    Object.keys(process.env)
      .filter((k) => k.startsWith("NEXT_PUBLIC_FORCE_FLAG_"))
      .forEach((k) => delete process.env[k]);
  });

  // The flag was created as literal `SHOW_ORG_SETTINGS` (in LaunchDarkly,
  // and kept as that key in PostHog) and cannot be renamed. The enum *value*
  // is the key the backend evaluates and serves on `/api/feature-flags` —
  // normalising it to the repo's usual kebab-case would resolve nothing and
  // pin the flag off forever.
  it("keeps the enum value pinned to the pre-existing flag key", () => {
    expect(Flag.SHOW_ORG_SETTINGS).toBe("SHOW_ORG_SETTINGS");
  });

  it("resolves SHOW_ORG_SETTINGS to false when the flag service has not answered", () => {
    const { result } = renderHook(() => useGetFlag(Flag.SHOW_ORG_SETTINGS));
    expect(result.current).toBe(false);
  });

  it("lets local dev / Playwright force the org UI on via the env override", () => {
    process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = "true";
    const { result } = renderHook(() => useGetFlag(Flag.SHOW_ORG_SETTINGS));
    expect(result.current).toBe(true);
  });

  it("still honours an explicit `false` override", () => {
    process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = "false";
    const { result } = renderHook(() => useGetFlag(Flag.SHOW_ORG_SETTINGS));
    expect(result.current).toBe(false);
  });
});
