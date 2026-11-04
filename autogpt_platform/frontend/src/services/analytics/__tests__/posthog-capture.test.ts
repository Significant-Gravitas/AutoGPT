import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { posthog } = vi.hoisted(() => ({
  posthog: {
    __loaded: false,
    capturing: false,
    // posthog-js returns the event it takes, and nothing when it drops one.
    capture: vi.fn((event: string): { event: string } | undefined => ({
      event,
    })),
    is_capturing() {
      return this.capturing;
    },
  },
}));

vi.mock("posthog-js", () => ({ default: posthog }));

import {
  capturePostHogEvent,
  resetPostHogCaptureQueueForTests,
} from "../posthog-capture";

const ONCE_KEY = "posthog_tour_started";

function storeConsentCookie(statistics: boolean) {
  const answer = `{necessary:true,preferences:false,statistics:${statistics},marketing:false}`;
  document.cookie = `CookieConsent=${encodeURIComponent(answer)}; Path=/`;
}

/** posthog.init has run and followAnalyticsConsent opted it in. */
function startCapturing() {
  posthog.__loaded = true;
  posthog.capturing = true;
}

function fire(event: string) {
  window.dispatchEvent(new Event(event));
}

describe("capturePostHogEvent", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    configureCookiebot();
    posthog.__loaded = false;
    posthog.capturing = false;
    posthog.capture.mockReset();
    sessionStorage.clear();
    resetPostHogCaptureQueueForTests();
  });

  afterEach(() => {
    resetPostHogCaptureQueueForTests();
    removeCookiebot();
    vi.unstubAllEnvs();
    vi.useRealTimers();
  });

  it("sends straight away while PostHog captures", () => {
    startCapturing();

    capturePostHogEvent("paywall_viewed", { surface: "billing" });

    expect(posthog.capture).toHaveBeenCalledWith("paywall_viewed", {
      surface: "billing",
    });
  });

  // A page's mount effect runs before the provider's init effect on a full
  // page load; posthog-js would silently drop the event.
  it("holds an event from a consented visitor until PostHog opts in, then sends it with its own time", () => {
    storeConsentCookie(true);
    const happenedAt = new Date("2026-09-23T10:00:00Z");
    vi.setSystemTime(happenedAt);

    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    // Loaded but still opted out: posthog-js would drop it.
    posthog.__loaded = true;
    vi.advanceTimersByTime(1_000);
    expect(posthog.capture).not.toHaveBeenCalled();
    expect(sessionStorage.getItem(ONCE_KEY)).toBeNull();

    posthog.capturing = true;
    vi.advanceTimersByTime(250);

    expect(posthog.capture).toHaveBeenCalledExactlyOnceWith(
      "tour_started",
      {},
      { timestamp: happenedAt },
    );
    expect(sessionStorage.getItem(ONCE_KEY)).toBe("1");
  });

  it("sends a held event once Cookiebot auto-consents after uc.js loads", () => {
    capturePostHogEvent("tour_started");
    vi.advanceTimersByTime(250);

    // A region that needs no consent: uc.js loads with an answer, no banner.
    installCookiebot({ statistics: true, marketing: true, preferences: true });
    fire("CookiebotOnConsentReady");
    startCapturing();
    vi.advanceTimersByTime(250);

    expect(posthog.capture).toHaveBeenCalledTimes(1);
    expect(posthog.capture.mock.calls[0][0]).toBe("tour_started");
  });

  // Between uc.js loading and its region check the API has no answer yet;
  // that is not the banner.
  it("keeps holding while uc.js is still deciding whether to ask", () => {
    capturePostHogEvent("tour_started");
    installCookiebot();
    vi.advanceTimersByTime(500);

    answerCookiebot({ statistics: true });
    startCapturing();
    vi.advanceTimersByTime(250);

    expect(posthog.capture).toHaveBeenCalledTimes(1);
  });

  it("drops the event of a visitor who declined analytics", () => {
    storeConsentCookie(false);
    startCapturing();
    posthog.capturing = false;

    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    vi.advanceTimersByTime(20_000);

    expect(posthog.capture).not.toHaveBeenCalled();
    // The guard is an analytics marker: nothing is stored without consent.
    expect(sessionStorage.getItem(ONCE_KEY)).toBeNull();
  });

  it("drops a declined visitor's event again on the next mount, still writing nothing", () => {
    storeConsentCookie(false);
    startCapturing();
    posthog.capturing = false;

    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    vi.advanceTimersByTime(20_000);

    expect(posthog.capture).not.toHaveBeenCalled();
    expect(sessionStorage.length).toBe(0);
  });

  it("sends a consented visitor's once-per-tab event only once", () => {
    storeConsentCookie(true);
    startCapturing();

    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });

    expect(posthog.capture).toHaveBeenCalledExactlyOnceWith("tour_started", {});
    expect(sessionStorage.getItem(ONCE_KEY)).toBe("1");
  });

  it("sends a dropped event on a later call once the visitor has consented in that tab", () => {
    installCookiebot();
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });

    answerCookiebot({ statistics: true });
    startCapturing();
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });

    expect(posthog.capture).toHaveBeenCalledExactlyOnceWith("tour_started", {});
    expect(sessionStorage.getItem(ONCE_KEY)).toBe("1");
  });

  it("drops held events when the visitor declines on load", () => {
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    installCookiebot();
    answerCookiebot({ statistics: false });

    startCapturing();
    vi.advanceTimersByTime(1_000);

    expect(posthog.capture).not.toHaveBeenCalled();
    expect(sessionStorage.getItem(ONCE_KEY)).toBeNull();
  });

  // Consent given on the banner covers what happens after it, never what
  // was held before it.
  it("drops held events when the banner asks, and does not replay them on accept", () => {
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    installCookiebot();
    fire("CookiebotOnDialogInit");

    expect(sessionStorage.getItem(ONCE_KEY)).toBeNull();
    answerCookiebot({ statistics: true });
    startCapturing();
    vi.advanceTimersByTime(1_000);

    expect(posthog.capture).not.toHaveBeenCalled();
  });

  it("drops held events once the banner has been up past the grace, even without its dialog event", () => {
    capturePostHogEvent("tour_started");
    installCookiebot();
    vi.advanceTimersByTime(2_000);

    answerCookiebot({ statistics: true });
    startCapturing();
    vi.advanceTimersByTime(1_000);

    expect(posthog.capture).not.toHaveBeenCalled();
  });

  it("drops an event captured while the banner is already asking", () => {
    installCookiebot();

    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    answerCookiebot({ statistics: true });
    startCapturing();
    vi.advanceTimersByTime(1_000);

    expect(posthog.capture).not.toHaveBeenCalled();
    expect(sessionStorage.getItem(ONCE_KEY)).toBeNull();
  });

  it("drops events when there is no consent manager to ask", () => {
    vi.unstubAllEnvs();
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");

    capturePostHogEvent("tour_started");
    startCapturing();
    posthog.capturing = false;
    vi.advanceTimersByTime(1_000);

    expect(posthog.capture).not.toHaveBeenCalled();
  });

  it("gives up on held events when nothing resolves", () => {
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });

    vi.advanceTimersByTime(60_000);
    storeConsentCookie(true);
    startCapturing();
    vi.advanceTimersByTime(1_000);

    expect(posthog.capture).not.toHaveBeenCalled();
    expect(sessionStorage.getItem(ONCE_KEY)).toBeNull();
  });

  it("sends a timed-out event on a later call once PostHog captures", () => {
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    vi.advanceTimersByTime(60_000);

    storeConsentCookie(true);
    startCapturing();
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });

    expect(posthog.capture).toHaveBeenCalledExactlyOnceWith("tour_started", {});
    expect(sessionStorage.getItem(ONCE_KEY)).toBe("1");
  });

  it("holds a once-per-tab event only once, and leaves its guard unset while held", () => {
    storeConsentCookie(true);

    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    expect(sessionStorage.getItem(ONCE_KEY)).toBeNull();

    startCapturing();
    vi.advanceTimersByTime(250);
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });

    expect(posthog.capture).toHaveBeenCalledTimes(1);
  });

  it("leaves the guard unset when posthog-js fails to send a held event", () => {
    storeConsentCookie(true);
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    posthog.capture.mockImplementation(() => {
      throw new Error("blocked");
    });

    startCapturing();
    vi.advanceTimersByTime(250);

    expect(posthog.capture).toHaveBeenCalledTimes(1);
    expect(sessionStorage.getItem(ONCE_KEY)).toBeNull();
  });

  it("leaves the guard unset when posthog-js fails to send straight away, so a later call sends", () => {
    startCapturing();
    posthog.capture.mockImplementationOnce(() => {
      throw new Error("blocked");
    });

    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    expect(sessionStorage.getItem(ONCE_KEY)).toBeNull();

    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    expect(posthog.capture).toHaveBeenCalledTimes(2);
    expect(sessionStorage.getItem(ONCE_KEY)).toBe("1");
  });

  // PostHog may still capture for a moment after consent is withdrawn; its
  // before_send hook then drops the event and capture returns nothing.
  it("leaves the guard unset when posthog-js drops the event itself", () => {
    startCapturing();
    posthog.capture.mockReturnValue(undefined);

    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });

    expect(posthog.capture).toHaveBeenCalledTimes(1);
    expect(sessionStorage.getItem(ONCE_KEY)).toBeNull();
  });

  it("leaves the guard unset when posthog-js drops a held event itself", () => {
    storeConsentCookie(true);
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    posthog.capture.mockReturnValue(undefined);

    startCapturing();
    vi.advanceTimersByTime(250);

    expect(posthog.capture).toHaveBeenCalledTimes(1);
    expect(sessionStorage.getItem(ONCE_KEY)).toBeNull();
  });

  it("never throws when posthog-js does", () => {
    startCapturing();
    posthog.capture.mockImplementation(() => {
      throw new Error("blocked");
    });

    expect(() => capturePostHogEvent("plan_selected")).not.toThrow();
  });
});
