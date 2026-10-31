import { beforeEach, describe, expect, it, vi } from "vitest";

const { posthog } = vi.hoisted(() => ({
  posthog: { __loaded: true, is_capturing: () => true, capture: vi.fn() },
}));
vi.mock("posthog-js", () => ({ default: posthog }));

import {
  buildStepLayout,
  NO_PAYWALL_STEPS,
  PAYWALL_LAST_STEPS,
} from "../store";
import { onboardingStepKey, trackOnboardingStep } from "../tracking";

describe("onboardingStepKey", () => {
  // Optional steps shift numeric positions; analytics keeps stable names.
  it("maps the same key across both step layouts", () => {
    expect(
      onboardingStepKey(PAYWALL_LAST_STEPS, PAYWALL_LAST_STEPS.painPoints),
    ).toBe("pain_points");
    expect(
      onboardingStepKey(NO_PAYWALL_STEPS, NO_PAYWALL_STEPS.painPoints),
    ).toBe("pain_points");
    expect(
      onboardingStepKey(PAYWALL_LAST_STEPS, PAYWALL_LAST_STEPS.preparing),
    ).toBe("preparing");
    expect(
      onboardingStepKey(NO_PAYWALL_STEPS, NO_PAYWALL_STEPS.preparing),
    ).toBe("preparing");
  });

  it("names the hire step when the layout has it", () => {
    const steps = buildStepLayout({
      hasPaywall: true,
    });
    expect(onboardingStepKey(steps, 5)).toBe("hire");
    expect(onboardingStepKey(steps, 6)).toBeNull();
    expect(onboardingStepKey(steps, 7)).toBe("preparing");
  });

  // The paywall reports itself as `paywall_viewed`; reporting it again here would
  // double-count the top of the funnel.
  it("returns null for the subscription step", () => {
    expect(
      onboardingStepKey(PAYWALL_LAST_STEPS, PAYWALL_LAST_STEPS.subscription!),
    ).toBeNull();
  });
});

describe("trackOnboardingStep", () => {
  beforeEach(() => {
    posthog.capture.mockReset();
    sessionStorage.clear();
  });

  it("reports each step to PostHog once, as onboarding_step_viewed", () => {
    trackOnboardingStep("role");
    trackOnboardingStep("role");
    trackOnboardingStep("hire");

    expect(posthog.capture.mock.calls).toEqual([
      ["onboarding_step_viewed", { step: "role" }],
      ["onboarding_step_viewed", { step: "hire" }],
    ]);
  });

  it("tracks the optional connection step once", () => {
    const steps = buildStepLayout({ hasConnect: true });
    const key = onboardingStepKey(steps, steps.connect!);
    expect(key).toBe("connect");
    if (key) {
      trackOnboardingStep(key);
      trackOnboardingStep(key);
    }
    expect(posthog.capture.mock.calls).toEqual([
      ["onboarding_step_viewed", { step: "connect" }],
    ]);
  });
});
