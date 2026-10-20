import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.hoisted(() => {
  process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND = "launchdarkly";
});
import { Flag, useGetFlag } from "../use-get-flag";

const vendor = vi.hoisted(() => ({
  configured: true,
  flags: {} as Record<string, unknown>,
}));

vi.mock("launchdarkly-react-client-sdk", () => ({
  useFlags: () => vendor.flags,
}));

vi.mock("@/services/environment", () => ({
  environment: { areFeatureFlagsEnabled: () => vendor.configured },
}));

const MATURE_FLAGS = [
  Flag.AUTOGPT_NEW_LAYOUT,
  Flag.CHAT_SEARCH,
  Flag.ARTIFACTS_PAGE,
  Flag.HIRE_EXPERTS,
  Flag.SKILLS_HUB,
  Flag.ONBOARDING_BRAIN_DUMP,
  Flag.ONBOARDING_EXPERT_TEAM,
  Flag.GRAPHITI_MEMORY,
  Flag.GRAPHITI_COMMUNITIES_ENABLED,
] as const;

const DEFAULT_SEARCH_TERMS = [
  "Research",
  "Marketing",
  "Product",
  "SEO",
  "Business",
];

beforeEach(() => {
  vendor.configured = true;
  vendor.flags = {};
  for (const key of Object.keys(process.env)) {
    if (key.startsWith("NEXT_PUBLIC_FORCE_FLAG_")) {
      vi.stubEnv(key, undefined);
    }
  }
});

afterEach(() => vi.unstubAllEnvs());

describe.each([true, false])("vendor configured: %s", (configured) => {
  beforeEach(() => {
    vendor.configured = configured;
  });

  it.each(MATURE_FLAGS)("defaults %s on without a vendor answer", (flag) => {
    expect(renderHook(() => useGetFlag(flag)).result.current).toBe(true);
  });

  it.each(
    Object.values(Flag).filter(
      (flag) =>
        !MATURE_FLAGS.some((mature) => mature === flag) &&
        flag !== Flag.MARKETPLACE_SEARCH_TERMS &&
        flag !== Flag.COPILOT_BOT_PLATFORMS &&
        flag !== Flag.COPILOT_STREAM_SHADOW,
    ),
  )("keeps the unselected %s off without a vendor answer", (flag) => {
    expect(renderHook(() => useGetFlag(flag)).result.current).toBe(false);
  });

  it.each(MATURE_FLAGS)("honors an explicit env opt-out for %s", (flag) => {
    vi.stubEnv(
      `NEXT_PUBLIC_FORCE_FLAG_${flag.replaceAll("-", "_").toUpperCase()}`,
      "false",
    );
    vendor.flags[flag] = true;

    expect(renderHook(() => useGetFlag(flag)).result.current).toBe(false);
  });

  it("uses the mature marketplace search terms without a vendor answer", () => {
    expect(
      renderHook(() => useGetFlag(Flag.MARKETPLACE_SEARCH_TERMS)).result
        .current,
    ).toEqual(DEFAULT_SEARCH_TERMS);
  });
});

describe("configured vendor answers", () => {
  it.each(MATURE_FLAGS)("honors a vendor opt-out for %s", (flag) => {
    vendor.flags[flag] = false;

    expect(renderHook(() => useGetFlag(flag)).result.current).toBe(false);
  });

  it("keeps the vendor's marketplace search terms", () => {
    vendor.flags[Flag.MARKETPLACE_SEARCH_TERMS] = ["Design", "Support"];

    expect(
      renderHook(() => useGetFlag(Flag.MARKETPLACE_SEARCH_TERMS)).result
        .current,
    ).toEqual(["Design", "Support"]);
  });

  it("ignores boolean env overrides for the search terms array", () => {
    vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_MARKETPLACE_SEARCH_TERMS", "false");

    expect(
      renderHook(() => useGetFlag(Flag.MARKETPLACE_SEARCH_TERMS)).result
        .current,
    ).toEqual(DEFAULT_SEARCH_TERMS);
  });

  it("still lets local dev explicitly enable an experimental gate", () => {
    vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_DREAM_PASS_ENABLED", "true");

    expect(
      renderHook(() => useGetFlag(Flag.DREAM_PASS_ENABLED)).result.current,
    ).toBe(true);
  });
});
