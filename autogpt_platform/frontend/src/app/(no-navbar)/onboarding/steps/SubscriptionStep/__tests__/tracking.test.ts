import { beforeEach, describe, expect, it, vi } from "vitest";

const { sendDatafastEvent } = vi.hoisted(() => ({
  sendDatafastEvent: vi.fn(),
}));

vi.mock("@/services/analytics", () => ({
  analytics: { sendDatafastEvent },
}));

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
    sendDatafastEvent.mockReset();
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

describe("trackPaywallView", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    sendDatafastEvent.mockReset();
    sessionStorage.clear();
  });

  it("reports the paywall impression", () => {
    trackPaywallView();

    expect(sendDatafastEvent).toHaveBeenCalledTimes(1);
    expect(sendDatafastEvent).toHaveBeenCalledWith("paywall_view", {});
  });

  // Cancelling Stripe checkout returns to `?step=1&subscription=cancelled` as a
  // full navigation, remounting the paywall. Without the session guard that
  // second mount would inflate the funnel's denominator.
  it("reports at most once per session", () => {
    trackPaywallView();
    trackPaywallView();

    expect(sendDatafastEvent).toHaveBeenCalledTimes(1);
  });

  it("still reports when sessionStorage is unavailable", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("storage blocked");
    });

    trackPaywallView();

    expect(sendDatafastEvent).toHaveBeenCalledTimes(1);
  });

  // This runs in a mount effect on the screen users pay from: if a third-party
  // script failure escaped, React would unmount the paywall into an error
  // boundary and there would be nothing to buy.
  it("never throws when the DataFast script does", () => {
    sendDatafastEvent.mockImplementation(() => {
      throw new Error("datafast unavailable");
    });

    expect(() => trackPaywallView()).not.toThrow();
  });
});
