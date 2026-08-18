import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { consent } from "@/services/consent/cookies";
import { analytics, flushDatafastQueue } from "./index";

vi.mock("@/services/consent/cookies", () => ({
  consent: { hasConsentFor: vi.fn() },
}));

function drainQueue() {
  window.datafast = vi.fn();
  flushDatafastQueue();
  delete window.datafast;
}

describe("sendDatafastEvent", () => {
  beforeEach(() => {
    drainQueue();
    vi.mocked(consent.hasConsentFor).mockReturnValue(true);
    window.history.pushState({}, "", "/");
  });

  afterEach(() => {
    drainQueue();
  });

  it("sends immediately when the DataFast script has loaded", () => {
    const datafast = vi.fn();
    window.datafast = datafast;

    analytics.sendDatafastEvent("tour_scenario_start", {
      scenario: "chat",
      step: 2,
    });

    expect(datafast).toHaveBeenCalledWith("tour_scenario_start", {
      scenario: "chat",
      step: 2,
    });
  });

  it("queues events fired before the script loads and flushes them in order", () => {
    analytics.sendDatafastEvent("tour_start", {});
    analytics.sendDatafastEvent("tour_scenario_start", { scenario: "x" });

    const datafast = vi.fn();
    window.datafast = datafast;
    flushDatafastQueue();

    expect(datafast.mock.calls).toEqual([
      ["tour_start", {}],
      ["tour_scenario_start", { scenario: "x" }],
    ]);
  });

  it("keeps events queued when flushing before the script loads", () => {
    analytics.sendDatafastEvent("tour_start", {});
    flushDatafastQueue();

    const datafast = vi.fn();
    window.datafast = datafast;
    flushDatafastQueue();

    expect(datafast).toHaveBeenCalledTimes(1);
    expect(datafast).toHaveBeenCalledWith("tour_start", {});
  });

  it("does not replay events after a flush", () => {
    analytics.sendDatafastEvent("tour_start", {});

    const datafast = vi.fn();
    window.datafast = datafast;
    flushDatafastQueue();
    flushDatafastQueue();

    expect(datafast).toHaveBeenCalledTimes(1);
  });

  it("flushes the backlog before sending when the script loads without onLoad", () => {
    analytics.sendDatafastEvent("tour_start", {});

    const datafast = vi.fn();
    window.datafast = datafast;
    analytics.sendDatafastEvent("tour_cta_click", { label: "pricing" });

    expect(datafast.mock.calls).toEqual([
      ["tour_start", {}],
      ["tour_cta_click", { label: "pricing" }],
    ]);
  });

  it("does not queue pre-consent events outside the tour", () => {
    vi.mocked(consent.hasConsentFor).mockReturnValue(false);

    analytics.sendDatafastEvent("run_agent", { agent_name: "x" });

    const datafast = vi.fn();
    window.datafast = datafast;
    flushDatafastQueue();

    expect(datafast).not.toHaveBeenCalled();
  });

  it("queues pre-consent events on the consent-exempt tour pages", () => {
    vi.mocked(consent.hasConsentFor).mockReturnValue(false);
    window.history.pushState({}, "", "/tour/chat");

    analytics.sendDatafastEvent("tour_start", {});

    const datafast = vi.fn();
    window.datafast = datafast;
    flushDatafastQueue();

    expect(datafast).toHaveBeenCalledWith("tour_start", {});
  });

  it("does not treat /tourism as a consent-exempt tour page", () => {
    vi.mocked(consent.hasConsentFor).mockReturnValue(false);
    window.history.pushState({}, "", "/tourism");

    analytics.sendDatafastEvent("tour_start", {});

    const datafast = vi.fn();
    window.datafast = datafast;
    flushDatafastQueue();

    expect(datafast).not.toHaveBeenCalled();
  });

  it("warns again when the queue overflows after a successful flush", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    for (let i = 0; i < 101; i++) {
      analytics.sendDatafastEvent("tour_start", {});
    }
    drainQueue();
    for (let i = 0; i < 101; i++) {
      analytics.sendDatafastEvent("tour_start", {});
    }

    expect(warn).toHaveBeenCalledTimes(2);
    warn.mockRestore();
  });

  it("caps the queue at 100, keeps the earliest events, and warns once", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    for (let i = 0; i < 150; i++) {
      analytics.sendDatafastEvent("tour_start", { i });
    }

    const datafast = vi.fn();
    window.datafast = datafast;
    flushDatafastQueue();

    expect(datafast.mock.calls.map(([, metadata]) => metadata)).toEqual(
      Array.from({ length: 100 }, (_, i) => ({ i })),
    );
    expect(warn).toHaveBeenCalledTimes(1);
    warn.mockRestore();
  });
});
