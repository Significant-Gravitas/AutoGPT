import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { posthog } = vi.hoisted(() => ({
  posthog: { __loaded: false, capture: vi.fn() },
}));

vi.mock("posthog-js", () => ({ default: posthog }));

import {
  capturePostHogEvent,
  resetPostHogCaptureQueueForTests,
} from "../posthog-capture";

describe("capturePostHogEvent", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    posthog.__loaded = false;
    posthog.capture.mockReset();
    resetPostHogCaptureQueueForTests();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("sends straight away once PostHog is initialised", () => {
    posthog.__loaded = true;

    capturePostHogEvent("paywall_viewed", { surface: "billing" });

    expect(posthog.capture).toHaveBeenCalledWith("paywall_viewed", {
      surface: "billing",
    });
  });

  // A page's mount effect runs before the provider's init effect on a full
  // page load; posthog-js would silently drop the event.
  it("holds an event sent before init and delivers it with its own time", () => {
    const happenedAt = new Date("2026-09-23T10:00:00Z");
    vi.setSystemTime(happenedAt);

    capturePostHogEvent("tour_started");
    expect(posthog.capture).not.toHaveBeenCalled();

    vi.setSystemTime(new Date("2026-09-23T10:00:01Z"));
    posthog.__loaded = true;
    vi.advanceTimersByTime(250);

    expect(posthog.capture).toHaveBeenCalledTimes(1);
    expect(posthog.capture).toHaveBeenCalledWith(
      "tour_started",
      {},
      { timestamp: happenedAt },
    );
  });

  it("gives up on held events when PostHog never initialises", () => {
    capturePostHogEvent("tour_started");

    vi.advanceTimersByTime(60_000);
    posthog.__loaded = true;
    vi.advanceTimersByTime(1_000);

    expect(posthog.capture).not.toHaveBeenCalled();
  });

  it("never throws when posthog-js does", () => {
    posthog.__loaded = true;
    posthog.capture.mockImplementation(() => {
      throw new Error("blocked");
    });

    expect(() => capturePostHogEvent("plan_selected")).not.toThrow();
  });
});
