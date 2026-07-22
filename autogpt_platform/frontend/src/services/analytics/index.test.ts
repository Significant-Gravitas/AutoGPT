import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { analytics, flushDatafastQueue } from "./index";

describe("sendDatafastEvent", () => {
  beforeEach(() => {
    delete window.datafast;
  });

  afterEach(() => {
    // Drain anything left queued so tests can't leak events into each other.
    window.datafast = vi.fn();
    flushDatafastQueue();
    delete window.datafast;
  });

  it("sends immediately when the DataFast script has loaded", () => {
    const datafast = vi.fn();
    window.datafast = datafast;

    analytics.sendDatafastEvent("tour_start", {});

    expect(datafast).toHaveBeenCalledWith("tour_start", {});
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

  it("caps the queue so a blocked script cannot grow it unbounded", () => {
    for (let i = 0; i < 150; i++) {
      analytics.sendDatafastEvent("tour_start", {});
    }

    const datafast = vi.fn();
    window.datafast = datafast;
    flushDatafastQueue();

    expect(datafast.mock.calls.length).toBeLessThan(150);
  });
});
