import { beforeEach, describe, expect, it, vi } from "vitest";

const { posthog } = vi.hoisted(() => ({
  posthog: { __loaded: true, is_capturing: () => true, capture: vi.fn() },
}));
vi.mock("posthog-js", () => ({ default: posthog }));

import {
  markPaywallCheckoutStarted,
  trackPaywallCheckoutCancelled,
  trackPaywallView,
} from "../tracking";

describe("PostHog paywall funnel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    posthog.capture.mockReset();
    sessionStorage.clear();
  });

  it("reports the onboarding paywall view once per tab", () => {
    trackPaywallView();
    trackPaywallView();

    expect(posthog.capture.mock.calls).toEqual([
      ["paywall_viewed", { surface: "onboarding" }],
    ]);
  });

  it("reports the return from Stripe without paying as checkout_abandoned", () => {
    trackPaywallCheckoutCancelled();
    trackPaywallCheckoutCancelled();

    expect(posthog.capture.mock.calls).toEqual([
      [
        "checkout_abandoned",
        { checkout_kind: "subscription", surface: "onboarding" },
      ],
    ]);
  });

  it("counts a second abandonment after a new checkout starts", () => {
    trackPaywallCheckoutCancelled();
    markPaywallCheckoutStarted();
    trackPaywallCheckoutCancelled();
    // The refresh after the second return is still guarded.
    trackPaywallCheckoutCancelled();

    expect(posthog.capture.mock.calls).toEqual([
      [
        "checkout_abandoned",
        { checkout_kind: "subscription", surface: "onboarding" },
      ],
      [
        "checkout_abandoned",
        { checkout_kind: "subscription", surface: "onboarding" },
      ],
    ]);
  });
});
