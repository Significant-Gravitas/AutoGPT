import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
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

const FORCE_FLAG_PREFIX = "NEXT_PUBLIC_FORCE_FLAG_";
let forcedFlagsBefore: Record<string, string | undefined> = {};

function clearForcedFlags() {
  const forced = Object.keys(process.env).filter((k) =>
    k.startsWith(FORCE_FLAG_PREFIX),
  );
  forcedFlagsBefore = Object.fromEntries(
    forced.map((k) => [k, process.env[k]]),
  );
  forced.forEach((k) => delete process.env[k]);
}

function restoreForcedFlags() {
  Object.keys(process.env)
    .filter((k) => k.startsWith(FORCE_FLAG_PREFIX))
    .forEach((k) => delete process.env[k]);
  Object.entries(forcedFlagsBefore).forEach(([k, v]) => {
    process.env[k] = v;
  });
}

const DREAM_GRAPHITI_FLAGS = [
  Flag.GRAPHITI_MEMORY,
  Flag.GRAPHITI_COMMUNITIES_ENABLED,
  Flag.DREAM_PASS_ENABLED,
  Flag.DREAM_PASS_WEB_FACT_CHECK,
  Flag.DREAM_PASS_INVALIDATE_ENTITY,
] as const;

describe("dream/graphiti flag defaults fail closed", () => {
  beforeEach(clearForcedFlags);
  afterEach(restoreForcedFlags);

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

// These two have been flipped to `true` in the defaults map twice now, and
// both times it silently swapped the onboarding wizard for every
// LaunchDarkly-less environment — local dev, CI, Playwright — which is how
// auth-happy-path stopped finding the pillbox step.
const ONBOARDING_FLAGS = [
  Flag.ONBOARDING_BRAIN_DUMP,
  Flag.AUTOGPT_NEW_LAYOUT,
] as const;

describe("expert flag default fails closed", () => {
  beforeEach(clearForcedFlags);
  afterEach(restoreForcedFlags);

  it("resolves HIRE_EXPERTS to false when LaunchDarkly has not answered, keeping the team surfaces hidden", () => {
    const { result } = renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));
    expect(result.current).toBe(false);
  });

  it("resolves EXPERT_TASK_MANAGEMENT to false when LaunchDarkly has not answered, keeping the task-receipt surfaces hidden", () => {
    const { result } = renderHook(() =>
      useGetFlag(Flag.EXPERT_TASK_MANAGEMENT),
    );
    expect(result.current).toBe(false);
  });
});

describe("onboarding flag defaults fail closed", () => {
  beforeEach(clearForcedFlags);
  afterEach(restoreForcedFlags);

  it.each(ONBOARDING_FLAGS)(
    "resolves %s to false when LaunchDarkly has not answered, so the default flow renders untouched",
    (flag) => {
      const { result } = renderHook(() => useGetFlag(flag));
      expect(result.current).toBe(false);
    },
  );

  it("still lets local dev force-enable the brain dump via the env override", () => {
    process.env.NEXT_PUBLIC_FORCE_FLAG_ONBOARDING_BRAIN_DUMP = "true";
    const { result } = renderHook(() => useGetFlag(Flag.ONBOARDING_BRAIN_DUMP));
    expect(result.current).toBe(true);
  });
});
