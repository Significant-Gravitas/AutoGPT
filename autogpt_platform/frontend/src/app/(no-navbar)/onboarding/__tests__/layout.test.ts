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

const layoutFlags = [
  Flag.ENABLE_PLATFORM_PAYMENT,
  Flag.ONBOARDING_BRAIN_DUMP,
  Flag.ONBOARDING_EXPERT_TEAM,
  Flag.HIRE_EXPERTS,
];
const initialProps = {
  userID: "first-user",
  isLoggedIn: true,
  isUserLoading: false,
  trialConfirmation: { ready: true, active: false },
};

beforeEach(() => {
  source.flags = Object.fromEntries(
    layoutFlags.map((flag) => [flag, { value: true, resolved: true }]),
  );
  source.local = false;
  source.paid = false;
});

describe("onboarding layout readiness", () => {
  it.each(layoutFlags)("waits for %s before freezing the layout", (flag) => {
    delete source.flags[flag];
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });

    expect(result.current.isReady).toBe(false);

    source.flags[flag] = { value: true, resolved: true };
    rerender(initialProps);

    expect(result.current.isReady).toBe(true);
    expect(result.current.isBrainDumpEnabled).toBe(true);
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

  it("keeps the active account's step order after flags change", () => {
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    const steps = result.current.steps;

    source.flags = Object.fromEntries(
      layoutFlags.map((flag) => [flag, { value: false, resolved: true }]),
    );
    rerender(initialProps);

    expect(result.current.isReady).toBe(true);
    expect(result.current.steps).toEqual(steps);
    expect(result.current.isBrainDumpEnabled).toBe(true);
  });

  it("uses the existing bounded fallback if a layout flag never resolves", () => {
    vi.useFakeTimers();
    try {
      delete source.flags[Flag.ONBOARDING_BRAIN_DUMP];
      const { result, unmount } = renderHook(useOnboardingLayout, {
        initialProps,
      });
      expect(result.current.isReady).toBe(false);

      act(() => vi.advanceTimersByTime(5000));

      expect(result.current.isReady).toBe(true);
      expect(result.current.isBrainDumpEnabled).toBe(false);
      expect(result.current.steps.hire).toBeUndefined();
      expect(result.current.steps.subscription).toBe(5);
      unmount();
    } finally {
      vi.useRealTimers();
    }
  });

  it("waits for the next account's flags instead of retaining the old layout", () => {
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    expect(result.current.steps.hire).toBe(5);

    source.flags = {};
    const nextUser = { ...initialProps, userID: "second-user" };
    rerender({ ...nextUser, isUserLoading: true });
    expect(result.current.isReady).toBe(false);

    source.flags[Flag.ENABLE_PLATFORM_PAYMENT] = {
      value: true,
      resolved: true,
    };
    rerender(nextUser);
    expect(result.current.isReady).toBe(false);

    source.flags[Flag.ONBOARDING_BRAIN_DUMP] = { value: true, resolved: true };
    source.flags[Flag.ONBOARDING_EXPERT_TEAM] = { value: true, resolved: true };
    source.flags[Flag.HIRE_EXPERTS] = { value: false, resolved: true };
    rerender(nextUser);

    expect(result.current.isReady).toBe(true);
    expect(result.current.isBrainDumpEnabled).toBe(true);
    expect(result.current.steps).toEqual({
      role: 1,
      painPoints: 2,
      subscription: 3,
      preparing: 4,
    });
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
