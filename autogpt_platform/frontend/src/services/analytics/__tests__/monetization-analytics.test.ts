import { beforeEach, describe, expect, it, vi } from "vitest";

const { posthog } = vi.hoisted(() => ({
  posthog: { __loaded: true, is_capturing: () => true, capture: vi.fn() },
}));

vi.mock("posthog-js", () => ({ default: posthog }));

import {
  markTrialCheckoutStarted,
  trackBillingPortalOpened,
  trackCheckoutAbandoned,
  trackPaywallViewed,
  trackPlanSelected,
  trackTrialCheckoutAbandoned,
} from "../monetization-analytics";

function eventsNamed(name: string) {
  return posthog.capture.mock.calls.filter(([event]) => event === name);
}

describe("monetization analytics", () => {
  beforeEach(() => {
    posthog.capture.mockReset();
    sessionStorage.clear();
  });

  it("reports a paywall view once per tab for each surface", () => {
    trackPaywallViewed("paywall_gate");
    trackPaywallViewed("paywall_gate");
    trackPaywallViewed("billing");

    expect(eventsNamed("paywall_viewed")).toEqual([
      ["paywall_viewed", { surface: "paywall_gate" }],
      ["paywall_viewed", { surface: "billing" }],
    ]);
  });

  it("reports the picked plan with its cycle, surface and pricing arm", () => {
    trackPlanSelected({
      subscription_tier: "MAX",
      billing_cycle: "yearly",
      surface: "onboarding",
      pricing_variant: "yearly-max",
    });

    expect(posthog.capture).toHaveBeenCalledWith("plan_selected", {
      subscription_tier: "MAX",
      billing_cycle: "yearly",
      surface: "onboarding",
      pricing_variant: "yearly-max",
    });
  });

  it("reports the billing portal and abandoned checkouts", () => {
    trackBillingPortalOpened("billing_payment_method");
    trackCheckoutAbandoned({ checkout_kind: "top_up", surface: "billing" });

    expect(posthog.capture).toHaveBeenCalledWith("billing_portal_opened", {
      surface: "billing_payment_method",
    });
    expect(posthog.capture).toHaveBeenCalledWith("checkout_abandoned", {
      checkout_kind: "top_up",
      surface: "billing",
    });
  });

  // `?trial=cancelled` stays in the URL, so a refresh must not count twice.
  it("reports an abandoned trial checkout once per tab", () => {
    trackTrialCheckoutAbandoned("billing");
    trackTrialCheckoutAbandoned("billing");

    expect(eventsNamed("checkout_abandoned")).toEqual([
      ["checkout_abandoned", { checkout_kind: "trial", surface: "billing" }],
    ]);
  });

  it("counts a second trial abandonment after a new checkout starts", () => {
    trackTrialCheckoutAbandoned("billing");
    markTrialCheckoutStarted("billing");
    trackTrialCheckoutAbandoned("billing");
    // The refresh after the second return is still guarded.
    trackTrialCheckoutAbandoned("billing");

    expect(eventsNamed("checkout_abandoned")).toEqual([
      ["checkout_abandoned", { checkout_kind: "trial", surface: "billing" }],
      ["checkout_abandoned", { checkout_kind: "trial", surface: "billing" }],
    ]);
  });
});
