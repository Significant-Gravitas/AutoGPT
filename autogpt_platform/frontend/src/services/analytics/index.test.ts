import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { hasConsentFor } from "@/services/consent/consent";
import { configureCookiebot } from "@/tests/integrations/cookiebot";
import {
  installGtagShim,
  removeGtagShim,
} from "@/tests/integrations/gtag-shim";
import { analytics, flushDatafastQueue } from "./index";

vi.mock("@/services/consent/consent", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/consent/consent")>()),
  hasConsentFor: vi.fn(),
}));

function drainQueue() {
  window.datafast = vi.fn();
  flushDatafastQueue();
  delete window.datafast;
}

describe("sendDatafastEvent", () => {
  beforeEach(() => {
    drainQueue();
    vi.mocked(hasConsentFor).mockReturnValue(true);
    window.history.pushState({}, "", "/");
  });

  afterEach(() => {
    drainQueue();
    vi.unstubAllEnvs();
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

  it.each([false, true])(
    "discards queued events after consent withdrawal (script loaded: %s)",
    (loaded) => {
      analytics.sendDatafastEvent("tour_start", {});
      vi.mocked(hasConsentFor).mockReturnValue(false);
      const datafast = vi.fn();
      if (loaded) window.datafast = datafast;

      flushDatafastQueue();
      expect(datafast).not.toHaveBeenCalled();

      vi.mocked(hasConsentFor).mockReturnValue(true);
      window.datafast = datafast;
      flushDatafastQueue();
      expect(datafast).not.toHaveBeenCalled();
    },
  );

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

  it("reports whether the event was sent, queued or dropped", () => {
    expect(analytics.sendDatafastEvent("tour_start", {})).toBe(true);

    window.datafast = vi.fn();
    expect(analytics.sendDatafastEvent("tour_start", {})).toBe(true);

    vi.mocked(hasConsentFor).mockReturnValue(false);
    expect(analytics.sendDatafastEvent("tour_start", {})).toBe(false);
  });

  it("does not queue pre-consent events", () => {
    vi.mocked(hasConsentFor).mockReturnValue(false);

    analytics.sendDatafastEvent("tour_cta_click", { label: "signup" });

    const datafast = vi.fn();
    window.datafast = datafast;
    flushDatafastQueue();

    expect(datafast).not.toHaveBeenCalled();
  });

  it("does not send without consent once the script has loaded", () => {
    // The script stays on window after the visitor takes consent back.
    vi.mocked(hasConsentFor).mockReturnValue(false);
    configureCookiebot();
    window.history.pushState({}, "", "/marketplace");
    const datafast = vi.fn();
    window.datafast = datafast;

    analytics.sendDatafastEvent("tour_cta_click", { label: "signup" });

    expect(datafast).not.toHaveBeenCalled();
  });

  it.each(["/tour", "/tour/chat"])(
    "waits for consent on the public tour (%s) like everywhere else",
    (path) => {
      vi.mocked(hasConsentFor).mockReturnValue(false);
      configureCookiebot();
      window.history.pushState({}, "", path);

      analytics.sendDatafastEvent("tour_start", {});

      const datafast = vi.fn();
      window.datafast = datafast;
      flushDatafastQueue();
      analytics.sendDatafastEvent("tour_scenario_start", { scenario: "x" });

      expect(datafast).not.toHaveBeenCalled();
    },
  );

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

describe("sendGAEvent", () => {
  afterEach(() => {
    removeGtagShim();
  });

  it("routes the command through the tag's own gtag shim", () => {
    // Regression: the arguments used to be spread into the dataLayer as a
    // plain array, and gtag.js only executes real `arguments` objects — so
    // every custom GA event was silently dropped.
    const calls = installGtagShim();

    analytics.sendGAEvent("event", "tour_start", { scenario: "chat" });

    expect(calls).toEqual([["event", "tour_start", { scenario: "chat" }]]);
  });

  it("drops the event when the tag never loaded", () => {
    removeGtagShim();

    expect(() =>
      analytics.sendGAEvent("event", "tour_start", {}),
    ).not.toThrow();
  });
});
