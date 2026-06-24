import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Flag, useGetFlag } from "../use-get-flag";

// LD is enabled but has not answered for any key — the exact window
// (init lag, outage, missing flag key) where useGetFlag falls through
// to defaultFlags in production.
vi.mock("launchdarkly-react-client-sdk", () => ({
  useFlags: () => ({}),
}));

vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));

vi.mock("@/services/environment", () => ({
  environment: { areFeatureFlagsEnabled: () => true },
}));

const DREAM_GRAPHITI_FLAGS = [
  Flag.GRAPHITI_MEMORY,
  Flag.GRAPHITI_COMMUNITIES_ENABLED,
  Flag.DREAM_PASS_ENABLED,
  Flag.DREAM_PASS_WEB_FACT_CHECK,
  Flag.DREAM_PASS_INVALIDATE_ENTITY,
] as const;

describe("dream/graphiti flag defaults fail closed", () => {
  beforeEach(() => {
    Object.keys(process.env)
      .filter((k) => k.startsWith("NEXT_PUBLIC_FORCE_FLAG_"))
      .forEach((k) => delete process.env[k]);
  });

  it.each(DREAM_GRAPHITI_FLAGS)(
    "resolves %s to false when LaunchDarkly has not answered, mirroring the backend's default=False gating",
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
