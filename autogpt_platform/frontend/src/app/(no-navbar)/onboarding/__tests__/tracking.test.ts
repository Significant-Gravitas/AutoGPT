import { beforeEach, describe, expect, it, vi } from "vitest";

const { sendDatafastEvent } = vi.hoisted(() => ({
  sendDatafastEvent: vi.fn(),
}));

vi.mock("@/services/analytics", () => ({
  analytics: { sendDatafastEvent },
}));

import { NO_PAYWALL_STEPS, PAYWALL_FIRST_STEPS } from "../store";
import { onboardingStepKey, trackOnboardingStep } from "../tracking";

describe("onboardingStepKey", () => {
  // Welcome is step 2 with the paywall first and step 1 without it, so the
  // number alone can't identify a step across cohorts.
  it("maps the same key across both step layouts", () => {
    expect(
      onboardingStepKey(PAYWALL_FIRST_STEPS, PAYWALL_FIRST_STEPS.welcome),
    ).toBe("welcome");
    expect(onboardingStepKey(NO_PAYWALL_STEPS, NO_PAYWALL_STEPS.welcome)).toBe(
      "welcome",
    );
    expect(
      onboardingStepKey(PAYWALL_FIRST_STEPS, PAYWALL_FIRST_STEPS.preparing),
    ).toBe("preparing");
    expect(
      onboardingStepKey(NO_PAYWALL_STEPS, NO_PAYWALL_STEPS.preparing),
    ).toBe("preparing");
  });

  // The paywall reports itself as `paywall_view`; reporting it again here would
  // double-count the top of the funnel.
  it("returns null for the subscription step", () => {
    expect(
      onboardingStepKey(PAYWALL_FIRST_STEPS, PAYWALL_FIRST_STEPS.subscription),
    ).toBeNull();
  });
});

describe("trackOnboardingStep", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    sendDatafastEvent.mockReset();
    sessionStorage.clear();
  });

  it("reports one goal per step name", () => {
    trackOnboardingStep("welcome");
    trackOnboardingStep("pain_points");

    expect(sendDatafastEvent).toHaveBeenNthCalledWith(
      1,
      "onboarding_welcome",
      {},
    );
    expect(sendDatafastEvent).toHaveBeenNthCalledWith(
      2,
      "onboarding_pain_points",
      {},
    );
  });

  it("reports each step at most once per session", () => {
    trackOnboardingStep("role");
    trackOnboardingStep("role");

    expect(sendDatafastEvent).toHaveBeenCalledTimes(1);
  });

  it("does not let one step suppress another", () => {
    trackOnboardingStep("role");
    trackOnboardingStep("preparing");

    expect(sendDatafastEvent).toHaveBeenCalledTimes(2);
  });

  it("still reports when sessionStorage is unavailable", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("storage blocked");
    });

    trackOnboardingStep("welcome");

    expect(sendDatafastEvent).toHaveBeenCalledTimes(1);
  });

  it("never throws when the DataFast script does", () => {
    sendDatafastEvent.mockImplementation(() => {
      throw new Error("datafast unavailable");
    });

    expect(() => trackOnboardingStep("welcome")).not.toThrow();
  });
});
