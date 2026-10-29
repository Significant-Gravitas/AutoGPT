import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Flag } from "@/services/feature-flags/use-get-flag";
import { useOnboardingLayout } from "../useOnboardingLayout";
const source = vi.hoisted(() => ({
  flags: {} as Record<string, { value: boolean; resolved: boolean }>,
  local: false,
  paid: false,
}));
vi.mock("@/services/feature-flags/flag-source", () => ({
  useFlagSource: (key: string) =>
    source.flags[key] ?? { value: undefined, resolved: false },
}));
vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));
vi.mock("@/services/environment", () => ({
  environment: {
    areFeatureFlagsEnabled: () => true,
    isLocal: () => source.local,
  },
}));
vi.mock("../usePaidCheckoutStatus", () => ({
  usePaidCheckoutStatus: () => ({
    active: source.paid,
    isLoading: false,
    ready: true,
  }),
}));
const initialProps = {
  userID: "first-user",
  isLoggedIn: true,
  isUserLoading: false,
  trialConfirmation: { ready: true, active: false },
};
beforeEach(() => {
  source.flags = {
    [Flag.ENABLE_PLATFORM_PAYMENT]: { value: true, resolved: true },
  };
  source.local = false;
  source.paid = false;
});
describe("onboarding layout", () => {
  it("includes team introductions, brain dump, and hiring without rollout flags", () => {
    const { result } = renderHook(useOnboardingLayout, { initialProps });
    expect(result.current.isReady).toBe(true);
    expect(result.current.steps).toEqual({
      team: 1,
      autopilot: 2,
      role: 3,
      painPoints: 4,
      hire: 5,
      subscription: 6,
      preparing: 7,
    });
  });
  it("waits only for payment before freezing checkout placement", () => {
    source.flags = {};
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    expect(result.current.isReady).toBe(false);
    source.flags[Flag.ENABLE_PLATFORM_PAYMENT] = {
      value: true,
      resolved: true,
    };
    rerender(initialProps);
    expect(result.current.isReady).toBe(true);
    expect(result.current.steps.subscription).toBe(6);
  });
  it("keeps the active account's payment order after a flag change", () => {
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    const steps = result.current.steps;
    source.flags[Flag.ENABLE_PLATFORM_PAYMENT] = {
      value: false,
      resolved: true,
    };
    rerender(initialProps);
    expect(result.current.steps).toEqual(steps);
  });
  it("uses the bounded payment fallback without dropping personalization", () => {
    vi.useFakeTimers();
    try {
      source.flags = {};
      const { result, unmount } = renderHook(useOnboardingLayout, {
        initialProps,
      });
      expect(result.current.isReady).toBe(false);
      act(() => vi.advanceTimersByTime(5000));
      expect(result.current.isReady).toBe(true);
      expect(result.current.steps).toMatchObject({
        team: 1,
        autopilot: 2,
        painPoints: 4,
        hire: 5,
        preparing: 6,
      });
      expect(result.current.steps.subscription).toBeUndefined();
      unmount();
    } finally {
      vi.useRealTimers();
    }
  });
  it("resolves the next account's checkout rather than retaining old payment state", () => {
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    source.flags = {};
    const nextUser = { ...initialProps, userID: "second-user" };
    rerender({ ...nextUser, isUserLoading: true });
    expect(result.current.isReady).toBe(false);
    source.flags[Flag.ENABLE_PLATFORM_PAYMENT] = {
      value: false,
      resolved: true,
    };
    rerender(nextUser);
    expect(result.current.isReady).toBe(true);
    expect(result.current.steps.hire).toBe(5);
    expect(result.current.steps.subscription).toBeUndefined();
  });
  it.each([
    ["unpaid hosted", false, false, false, true],
    ["paid hosted", false, true, false, false],
    ["active trial", false, false, true, false],
    ["self-hosted", true, false, false, false],
  ] as const)(
    "preserves the %s checkout layout",
    (_name, local, paid, trial, paywall) => {
      source.local = local;
      source.paid = paid;
      source.flags[Flag.ENABLE_PLATFORM_PAYMENT].value = !local;
      const { result } = renderHook(useOnboardingLayout, {
        initialProps: {
          ...initialProps,
          trialConfirmation: { ready: true, active: trial },
        },
      });
      expect(result.current.isReady).toBe(true);
      expect(result.current.isPaymentEnabled).toBe(paywall);
      expect(result.current.steps.subscription !== undefined).toBe(paywall);
      expect(result.current.steps.connect !== undefined).toBe(local);
      expect(result.current.steps.hire).toBe(5);
    },
  );
});
