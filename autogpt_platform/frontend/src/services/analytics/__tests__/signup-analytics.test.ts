import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { posthog, consent } = vi.hoisted(() => ({
  posthog: {
    __loaded: true,
    is_capturing: vi.fn(() => true),
    capture: vi.fn(),
  },
  consent: {
    answer: null as { analytics: boolean } | null,
    listeners: [] as (() => void)[],
  },
}));

vi.mock("posthog-js", () => ({ default: posthog }));

vi.mock("@/services/consent/consent", () => ({
  getConsentAnswer: () => consent.answer,
  isAwaitingConsentAnswer: () => false,
  isConsentManagerConfigured: () => true,
  subscribeToConsent: (listener: () => void) => {
    consent.listeners.push(listener);
    return () => undefined;
  },
  subscribeToConsentPrompt: () => () => undefined,
}));

import { resetPostHogCaptureQueueForTests } from "../posthog-capture";
import { trackSignupMarketingOptOut } from "../signup-analytics";

beforeEach(() => {
  posthog.capture.mockReset();
  posthog.is_capturing.mockReturnValue(true);
  consent.answer = null;
  consent.listeners = [];
});

afterEach(() => {
  resetPostHogCaptureQueueForTests();
});

describe("trackSignupMarketingOptOut", () => {
  it("reports the opt-out with only its surface", () => {
    trackSignupMarketingOptOut();

    expect(posthog.capture.mock.calls).toEqual([
      ["marketing_opted_out", { surface: "signup" }],
    ]);
  });

  it("holds the opt-out until analytics consent is given, then sends it", () => {
    posthog.is_capturing.mockReturnValue(false);
    consent.answer = { analytics: true };

    trackSignupMarketingOptOut();
    expect(posthog.capture).not.toHaveBeenCalled();

    posthog.is_capturing.mockReturnValue(true);
    consent.listeners.forEach((listener) => listener());

    expect(posthog.capture).toHaveBeenCalledTimes(1);
    expect(posthog.capture.mock.calls[0].slice(0, 2)).toEqual([
      "marketing_opted_out",
      { surface: "signup" },
    ]);
  });

  it("swallows a blocked analytics host rather than breaking signup", () => {
    posthog.capture.mockImplementation(() => {
      throw new Error("blocked by client");
    });

    expect(() => trackSignupMarketingOptOut()).not.toThrow();
  });
});
