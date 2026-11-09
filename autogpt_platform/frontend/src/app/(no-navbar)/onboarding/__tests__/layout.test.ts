import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useOnboardingLayout } from "../useOnboardingLayout";
const source = vi.hoisted(() => ({
  payment: { isPaymentEnabled: true, isResolved: true, isError: false },
  local: false,
  paid: false,
  interview: { enabled: false, ready: true },
}));
vi.mock("@/hooks/usePlatformPayment", () => ({
  usePlatformPayment: () => source.payment,
}));
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ONBOARDING_INTERVIEW: "onboarding-interview" },
  useFlagStatus: () => ({ ...source.interview, answered: true }),
}));
vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));
vi.mock("@/services/environment", () => ({
  environment: {
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
  source.payment = { isPaymentEnabled: true, isResolved: true, isError: false };
  source.local = false;
  source.paid = false;
  source.interview = { enabled: false, ready: true };
});
describe("onboarding layout", () => {
  it("lays the interview out in the brain dump's place when its flag is on", () => {
    source.interview = { enabled: true, ready: true };

    const { result } = renderHook(useOnboardingLayout, { initialProps });

    expect(result.current.steps).toEqual({
      team: 1,
      autopilot: 2,
      role: 3,
      interview: 4,
      subscription: 5,
      preparing: 6,
    });
  });

  it("waits for the interview flag before freezing the layout", () => {
    source.interview = { enabled: false, ready: false };
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    expect(result.current.isReady).toBe(false);

    source.interview = { enabled: true, ready: true };
    rerender(initialProps);

    expect(result.current.isReady).toBe(true);
    expect(result.current.steps.interview).toBe(4);
    expect(result.current.steps.painPoints).toBeUndefined();
  });

  it("includes team introductions, brain dump, and the plan + trial step without rollout flags", () => {
    const { result } = renderHook(useOnboardingLayout, { initialProps });
    expect(result.current.isReady).toBe(true);
    expect(result.current.steps).toEqual({
      team: 1,
      autopilot: 2,
      role: 3,
      painPoints: 4,
      subscription: 5,
      preparing: 6,
    });
  });
  it("waits only for payment before freezing checkout placement", () => {
    source.payment = {
      isPaymentEnabled: false,
      isResolved: false,
      isError: false,
    };
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    expect(result.current.isReady).toBe(false);
    source.payment = {
      isPaymentEnabled: true,
      isResolved: true,
      isError: false,
    };
    rerender(initialProps);
    expect(result.current.isReady).toBe(true);
    expect(result.current.steps.subscription).toBe(5);
  });
  it("lays out a failed read as payment off without freezing it", () => {
    source.payment = {
      isPaymentEnabled: false,
      isResolved: true,
      isError: true,
    };
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    expect(result.current.isReady).toBe(true);
    expect(result.current.isPaymentEnabled).toBe(false);
    source.payment = {
      isPaymentEnabled: true,
      isResolved: true,
      isError: false,
    };
    rerender(initialProps);
    expect(result.current.isPaymentEnabled).toBe(true);
    expect(result.current.steps.subscription).toBe(5);
  });
  it("keeps the active account's payment order after a setting change", () => {
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    const steps = result.current.steps;
    source.payment = {
      isPaymentEnabled: false,
      isResolved: true,
      isError: false,
    };
    rerender(initialProps);
    expect(result.current.steps).toEqual(steps);
  });
  it("resolves the next account's checkout rather than retaining old payment state", () => {
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    source.payment = {
      isPaymentEnabled: false,
      isResolved: false,
      isError: false,
    };
    const nextUser = { ...initialProps, userID: "second-user" };
    rerender({ ...nextUser, isUserLoading: true });
    expect(result.current.isReady).toBe(false);
    source.payment = {
      isPaymentEnabled: false,
      isResolved: true,
      isError: false,
    };
    rerender(nextUser);
    expect(result.current.isReady).toBe(true);
    expect(result.current.isPaymentEnabled).toBe(false);
    // No paywall configured: the plan stands alone where the paywall was.
    expect(result.current.steps.subscription).toBe(5);
    expect(result.current.steps.hire).toBeUndefined();
  });
  it.each([
    ["unpaid hosted", false, false, false, true, true],
    ["paid hosted", false, true, false, false, false],
    ["active trial", false, false, true, false, false],
    ["self-hosted", true, false, false, false, true],
  ] as const)(
    "preserves the %s checkout layout",
    (_name, local, paid, trial, paywall, planStep) => {
      source.local = local;
      source.paid = paid;
      source.payment = {
        isPaymentEnabled: !local,
        isResolved: true,
        isError: false,
      };
      const { result } = renderHook(useOnboardingLayout, {
        initialProps: {
          ...initialProps,
          trialConfirmation: { ready: true, active: trial },
        },
      });
      expect(result.current.isReady).toBe(true);
      expect(result.current.isPaymentEnabled).toBe(paywall);
      expect(result.current.steps.subscription !== undefined).toBe(planStep);
      expect(result.current.steps.connect !== undefined).toBe(local);
      expect(result.current.steps.hire).toBeUndefined();
    },
  );
});
