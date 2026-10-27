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

  // The plan step keeps the paywall's key with or without a paywall, so one
  // funnel reads every cohort; the hire step is gone from every layout.
  it("names the plan step subscription in every layout", () => {
    for (const steps of [
      PAYWALL_LAST_STEPS,
      NO_PAYWALL_STEPS,
      buildStepLayout({ hasPlan: true, hasConnect: true }),
    ]) {
      expect(onboardingStepKey(steps, steps.subscription!)).toBe(
        "subscription",
      );
    }
    expect(onboardingStepKey(PAYWALL_LAST_STEPS, 6)).toBe("preparing");
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
    trackOnboardingStep("subscription");

    expect(posthog.capture.mock.calls).toEqual([
      ["onboarding_step_viewed", { step: "role" }],
      ["onboarding_step_viewed", { step: "subscription" }],
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
