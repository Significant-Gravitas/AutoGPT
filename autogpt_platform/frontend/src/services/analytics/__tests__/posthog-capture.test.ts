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
    capture: vi.fn(),
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
    // Deliberately dropped: the tab does not try again.
    expect(sessionStorage.getItem(ONCE_KEY)).toBe("1");
  });

  it("drops held events when the visitor declines on load", () => {
    capturePostHogEvent("tour_started");
    installCookiebot();
    answerCookiebot({ statistics: false });

    startCapturing();
    vi.advanceTimersByTime(1_000);

    expect(posthog.capture).not.toHaveBeenCalled();
  });

  // Consent given on the banner covers what happens after it, never what
  // was held before it.
  it("drops held events when the banner asks, and does not replay them on accept", () => {
    capturePostHogEvent("tour_started", {}, { oncePerTabKey: ONCE_KEY });
    installCookiebot();
    fire("CookiebotOnDialogInit");

    expect(sessionStorage.getItem(ONCE_KEY)).toBe("1");
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
    expect(sessionStorage.getItem(ONCE_KEY)).toBe("1");
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

  it("never throws when posthog-js does", () => {
    startCapturing();
    posthog.capture.mockImplementation(() => {
      throw new Error("blocked");
    });

    expect(() => capturePostHogEvent("plan_selected")).not.toThrow();
  });
});
