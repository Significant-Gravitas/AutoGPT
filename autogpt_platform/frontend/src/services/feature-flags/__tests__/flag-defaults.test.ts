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

// A closed account is only erased once SECRT-2854's routine runs, so a `true`
// here would let every LaunchDarkly-less environment delete accounts that
// nothing ever erases.
describe("self-serve account deletion flag fails closed", () => {
  beforeEach(() => {
    Object.keys(process.env)
      .filter((k) => k.startsWith("NEXT_PUBLIC_FORCE_FLAG_"))
      .forEach((k) => delete process.env[k]);
  });

  it("uses the backend's key", () => {
    expect(Flag.SELF_SERVE_ACCOUNT_DELETION).toBe(
      "self-serve-account-deletion",
    );
  });

  it("resolves to false when LaunchDarkly has not answered", () => {
    const { result } = renderHook(() =>
      useGetFlag(Flag.SELF_SERVE_ACCOUNT_DELETION),
    );
    expect(result.current).toBe(false);
  });

  it("still lets local dev force it on via the env override", () => {
    process.env.NEXT_PUBLIC_FORCE_FLAG_SELF_SERVE_ACCOUNT_DELETION = "true";
    const { result } = renderHook(() =>
      useGetFlag(Flag.SELF_SERVE_ACCOUNT_DELETION),
    );
    expect(result.current).toBe(true);
  });
});

describe("the onboarding voice interview flag fails closed", () => {
  const ENV_KEY = "NEXT_PUBLIC_FORCE_FLAG_ONBOARDING_INTERVIEW_VOICE";

  beforeEach(() => {
    delete process.env[ENV_KEY];
  });

  it("stays off when the flag service has not answered, as the call start route does", () => {
    const { result } = renderHook(() =>
      useGetFlag(Flag.ONBOARDING_INTERVIEW_VOICE),
    );
    expect(result.current).toBe(false);
  });

  it("can be turned on for a local build", () => {
    process.env[ENV_KEY] = "true";
    const { result } = renderHook(() =>
      useGetFlag(Flag.ONBOARDING_INTERVIEW_VOICE),
    );
    expect(result.current).toBe(true);
  });
});
