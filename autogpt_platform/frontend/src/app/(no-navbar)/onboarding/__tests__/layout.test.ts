import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Flag } from "@/services/feature-flags/use-get-flag";
import {
  INTERVIEW_FLAG_WAIT_MS,
  useOnboardingLayout,
} from "../useOnboardingLayout";
const source = vi.hoisted(() => ({
  payment: { isPaymentEnabled: true, isResolved: true, isError: false },
  flags: {} as Record<string, { value: unknown; resolved: boolean }>,
  local: false,
  paid: false,
  signIn: { provider: null as string | null, isReady: true },
}));
vi.mock("@/hooks/usePlatformPayment", () => ({
  usePlatformPayment: () => source.payment,
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
    isLocal: () => source.local,
  },
}));
vi.mock("../steps/AccessStep/useSignInProvider", () => ({
  useSignInProvider: () => source.signIn,
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
// The step-order flag answers with no experiment, so the default order holds.
const DEFAULT_ORDER = { value: null, resolved: true };
// The interview flag answers off, so the brain dump holds its place.
const NO_INTERVIEW = { value: false, resolved: true };
const INTERVIEW = { value: true, resolved: true };
beforeEach(() => {
  source.payment = { isPaymentEnabled: true, isResolved: true, isError: false };
  source.flags = {
    [Flag.ONBOARDING_STEP_ORDER]: DEFAULT_ORDER,
    [Flag.ONBOARDING_INTERVIEW]: NO_INTERVIEW,
  };
  source.local = false;
  source.paid = false;
  source.signIn = { provider: null, isReady: true };
});
afterEach(() => {
  vi.useRealTimers();
});
describe("onboarding layout", () => {
  it("lays the interview out in the brain dump's place when its flag is on", () => {
    source.flags[Flag.ONBOARDING_INTERVIEW] = INTERVIEW;

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
    delete source.flags[Flag.ONBOARDING_INTERVIEW];
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    expect(result.current.isReady).toBe(false);

    source.flags[Flag.ONBOARDING_INTERVIEW] = INTERVIEW;
    rerender(initialProps);

    expect(result.current.isReady).toBe(true);
    expect(result.current.steps.interview).toBe(4);
    expect(result.current.steps.painPoints).toBeUndefined();
  });

  it("waits past the flag's own timeout for its answer, and lays out the interview it gives", () => {
    vi.useFakeTimers();
    delete source.flags[Flag.ONBOARDING_INTERVIEW];
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    // Past the flag's own 5 s timeout, which serves its default.
    act(() => vi.advanceTimersByTime(5_500));
    expect(result.current.isReady).toBe(false);

    source.flags[Flag.ONBOARDING_INTERVIEW] = INTERVIEW;
    rerender(initialProps);

    expect(result.current.isReady).toBe(true);
    expect(result.current.steps.interview).toBe(4);
    expect(result.current.isLayoutAnswered).toBe(true);
  });

  it("lays the wizard out without the interview for this visit once its flag has not answered in time", () => {
    vi.useFakeTimers();
    delete source.flags[Flag.ONBOARDING_INTERVIEW];
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    act(() => vi.advanceTimersByTime(INTERVIEW_FLAG_WAIT_MS - 1));
    expect(result.current.isReady).toBe(false);

    act(() => vi.advanceTimersByTime(1));

    expect(result.current.isReady).toBe(true);
    expect(result.current.steps.painPoints).toBe(4);
    expect(result.current.steps.interview).toBeUndefined();
    expect(result.current.isLayoutAnswered).toBe(false);
    // An answer after the wait does not reorder the wizard under the person.
    source.flags[Flag.ONBOARDING_INTERVIEW] = INTERVIEW;
    rerender(initialProps);
    expect(result.current.steps.interview).toBeUndefined();
    expect(result.current.isLayoutAnswered).toBe(false);
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
  it("waits only for payment and the step order before freezing checkout placement", () => {
    source.payment = {
      isPaymentEnabled: false,
      isResolved: false,
      isError: false,
    };
    source.flags = { [Flag.ONBOARDING_INTERVIEW]: NO_INTERVIEW };
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    expect(result.current.isReady).toBe(false);
    source.payment = {
      isPaymentEnabled: true,
      isResolved: true,
      isError: false,
    };
    source.flags[Flag.ONBOARDING_STEP_ORDER] = DEFAULT_ORDER;
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
  it("offers a Microsoft sign-up its connection before anything else", () => {
    source.payment = {
      isPaymentEnabled: false,
      isResolved: true,
      isError: false,
    };
    source.signIn = { provider: "microsoft", isReady: true };
    const { result } = renderHook(useOnboardingLayout, { initialProps });
    expect(result.current.steps).toEqual({
      access: 1,
      team: 2,
      autopilot: 3,
      role: 4,
      painPoints: 5,
      subscription: 6,
      preparing: 7,
    });
  });
  it("offers the connection after the paywall when connecting needs a plan", () => {
    source.signIn = { provider: "microsoft", isReady: true };
    source.flags[Flag.ONBOARDING_STEP_ORDER] = {
      value: ["access"],
      resolved: true,
    };
    const { result } = renderHook(useOnboardingLayout, { initialProps });
    expect(result.current.steps).toEqual({
      team: 1,
      autopilot: 2,
      role: 3,
      painPoints: 4,
      subscription: 5,
      access: 6,
      preparing: 7,
    });
  });
  it("doesn't freeze the connection's place while payment is unknown", () => {
    source.payment = {
      isPaymentEnabled: false,
      isResolved: true,
      isError: true,
    };
    source.signIn = { provider: "microsoft", isReady: true };
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    expect(result.current.isReady).toBe(true);
    expect(result.current.steps.access).toBe(1);
    source.payment = {
      isPaymentEnabled: true,
      isResolved: true,
      isError: false,
    };
    rerender(initialProps);
    // Connecting needs a plan once payment is known to be on.
    expect(result.current.steps.subscription).toBe(5);
    expect(result.current.steps.access).toBe(6);
  });
  it("keeps a late step-order answer from moving steps while payment is unknown", () => {
    source.payment = {
      isPaymentEnabled: false,
      isResolved: true,
      isError: true,
    };
    source.signIn = { provider: "microsoft", isReady: true };
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    const steps = result.current.steps;
    source.flags[Flag.ONBOARDING_STEP_ORDER] = {
      value: ["team", "autopilot", "role", "access"],
      resolved: true,
    };
    rerender(initialProps);
    expect(result.current.steps).toEqual(steps);
  });
  it("waits for the sign-in lookup before freezing the steps", () => {
    source.payment = {
      isPaymentEnabled: false,
      isResolved: true,
      isError: false,
    };
    source.signIn = { provider: null, isReady: false };
    const { result, rerender } = renderHook(useOnboardingLayout, {
      initialProps,
    });
    expect(result.current.isReady).toBe(false);
    source.signIn = { provider: "microsoft", isReady: true };
    rerender(initialProps);
    expect(result.current.isReady).toBe(true);
    expect(result.current.steps.access).toBe(1);
  });
  it("lets an experiment move the access step", () => {
    source.payment = {
      isPaymentEnabled: false,
      isResolved: true,
      isError: false,
    };
    source.signIn = { provider: "microsoft", isReady: true };
    source.flags[Flag.ONBOARDING_STEP_ORDER] = {
      value: ["team", "autopilot", "role", "access"],
      resolved: true,
    };
    const { result } = renderHook(useOnboardingLayout, { initialProps });
    expect(result.current.steps).toEqual({
      team: 1,
      autopilot: 2,
      role: 3,
      access: 4,
      painPoints: 5,
      subscription: 6,
      preparing: 7,
    });
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
