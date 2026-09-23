import { afterEach, describe, expect, it, vi } from "vitest";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import {
  getConsent,
  getConsentAnswer,
  getConsentManagerStatus,
  hasConsentFor,
  isConsentManagerConfigured,
  NO_CONSENT,
  openConsentSettings,
  subscribeToConsent,
  subscribeToConsentManagerStatus,
  toConsentState,
} from "./consent";
import { readConsentFromCookieHeader } from "./consent-server";

const STATISTICS_ONLY =
  "{stamp:'abc',necessary:true,preferences:false,statistics:true,marketing:false,method:'explicit',ver:1,utc:1,region:'de'}";

const STATISTICS_DENIED = STATISTICS_ONLY.replace(
  "statistics:true",
  "statistics:false",
);

function storeCookie(value: string) {
  document.cookie = `CookieConsent=${encodeURIComponent(value)}; Path=/`;
}

afterEach(() => {
  removeCookiebot();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("category mapping", () => {
  it("maps analytics and monitoring to statistics, advertising to marketing", () => {
    expect(
      toConsentState({
        necessary: true,
        preferences: true,
        statistics: true,
        marketing: false,
      }),
    ).toEqual({ analytics: true, monitoring: true, advertising: false });
    expect(
      toConsentState({
        necessary: true,
        preferences: true,
        statistics: false,
        marketing: true,
      }),
    ).toEqual({ analytics: false, monitoring: false, advertising: true });
  });

  it("denies everything without an answer", () => {
    expect(toConsentState(null)).toEqual(NO_CONSENT);
  });
});

describe("without a Cookiebot domain group", () => {
  it("denies every category, whatever the browser holds", () => {
    installCookiebot({ statistics: true, marketing: true });
    storeCookie("-1");

    expect(isConsentManagerConfigured()).toBe(false);
    expect(getConsent()).toEqual(NO_CONSENT);
    expect(hasConsentFor("analytics")).toBe(false);
  });

  it("denies every category on the server too", () => {
    expect(readConsentFromCookieHeader("CookieConsent=-1")).toEqual(NO_CONSENT);
  });
});

describe("with a Cookiebot domain group", () => {
  it("reads a stored answer before Cookiebot has loaded", () => {
    configureCookiebot();
    storeCookie(STATISTICS_ONLY);

    expect(getConsent()).toEqual({
      analytics: true,
      monitoring: true,
      advertising: false,
    });
  });

  it("follows the loaded script once the visitor has answered", () => {
    configureCookiebot();
    storeCookie(STATISTICS_ONLY);
    installCookiebot({ statistics: false, marketing: true });

    expect(getConsent()).toEqual({
      analytics: false,
      monitoring: false,
      advertising: true,
    });
  });

  it("denies everything until the visitor answers", () => {
    configureCookiebot();
    installCookiebot();

    expect(getConsent()).toEqual(NO_CONSENT);
    expect(getConsentAnswer()).toBeNull();
  });

  it("ignores a stored grant the loaded script no longer accepts", () => {
    // Cookiebot withdraws an answer after a banner version bump, an expiry
    // or a region change and asks again, leaving the old cookie in place
    // until the visitor answers.
    configureCookiebot();
    storeCookie(STATISTICS_ONLY);
    installCookiebot();

    expect(getConsent()).toEqual(NO_CONSENT);
    expect(getConsentAnswer()).toBeNull();
    expect(hasConsentFor("analytics")).toBe(false);
  });

  it("falls back to the stored answer only while the script is absent", () => {
    configureCookiebot();
    storeCookie(STATISTICS_ONLY);

    expect(hasConsentFor("analytics")).toBe(true);

    installCookiebot();
    expect(hasConsentFor("analytics")).toBe(false);

    removeCookiebot();
    storeCookie(STATISTICS_ONLY);
    expect(hasConsentFor("analytics")).toBe(true);
  });

  it("reports the answer, or null when there is none", () => {
    configureCookiebot();
    expect(getConsentAnswer()).toBeNull();

    storeCookie(STATISTICS_DENIED);
    expect(getConsentAnswer()).toEqual(NO_CONSENT);

    installCookiebot({ marketing: true });
    expect(getConsentAnswer()).toEqual({
      analytics: false,
      monitoring: false,
      advertising: true,
    });
  });

  it("notifies subscribers on accept and decline, and stops after unsubscribe", () => {
    configureCookiebot();
    installCookiebot();
    const listener = vi.fn();
    const unsubscribe = subscribeToConsent(listener);

    answerCookiebot({ statistics: true });
    expect(listener).toHaveBeenCalledTimes(1);
    expect(hasConsentFor("analytics")).toBe(true);

    answerCookiebot({});
    expect(listener).toHaveBeenCalledTimes(2);
    expect(hasConsentFor("analytics")).toBe(false);

    unsubscribe();
    answerCookiebot({ statistics: true });
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it("opens the Cookiebot dialog to change the answer", () => {
    configureCookiebot();
    const { renew } = installCookiebot();

    openConsentSettings();

    expect(renew).toHaveBeenCalledOnce();
  });

  it("does nothing when the Cookiebot script never loaded", () => {
    configureCookiebot();

    expect(() => openConsentSettings()).not.toThrow();
  });

  it("reads the request cookie on the server", () => {
    configureCookiebot();
    const header = `theme=dark; CookieConsent=${encodeURIComponent(STATISTICS_ONLY)}`;

    expect(readConsentFromCookieHeader(header)).toEqual({
      analytics: true,
      monitoring: true,
      advertising: false,
    });
    expect(readConsentFromCookieHeader("theme=dark")).toEqual(NO_CONSENT);
    expect(readConsentFromCookieHeader(null)).toEqual(NO_CONSENT);
  });

  it.each([
    ["the granting copy first", [STATISTICS_ONLY, STATISTICS_DENIED]],
    ["the granting copy last", [STATISTICS_DENIED, STATISTICS_ONLY]],
  ])(
    "denies on client and server alike when duplicate cookies disagree (%s)",
    (_, [first, second]) => {
      configureCookiebot();
      const header = `CookieConsent=${encodeURIComponent(first)}; CookieConsent=${encodeURIComponent(second)}`;
      vi.spyOn(document, "cookie", "get").mockReturnValue(header);

      expect(getConsent()).toEqual(NO_CONSENT);
      expect(readConsentFromCookieHeader(header)).toEqual(NO_CONSENT);
    },
  );

  it("grants what duplicate cookies agree on", () => {
    configureCookiebot();
    const header = `CookieConsent=${encodeURIComponent(STATISTICS_ONLY)}; CookieConsent=-1`;
    vi.spyOn(document, "cookie", "get").mockReturnValue(header);

    expect(getConsent()).toEqual({
      analytics: true,
      monitoring: true,
      advertising: false,
    });
    expect(readConsentFromCookieHeader(header)).toEqual(getConsent());
  });
});

describe("consent manager status", () => {
  it("is ready once the Cookiebot script has loaded", () => {
    configureCookiebot();
    installCookiebot();

    expect(getConsentManagerStatus()).toBe("ready");
  });

  it("is unavailable when the page finished loading without it", () => {
    configureCookiebot();
    vi.spyOn(document, "readyState", "get").mockReturnValue("complete");

    expect(getConsentManagerStatus()).toBe("unavailable");
  });

  it("is still loading while the page is", () => {
    configureCookiebot();
    vi.spyOn(document, "readyState", "get").mockReturnValue("interactive");

    expect(getConsentManagerStatus()).toBe("loading");
  });

  it("notifies when the Cookiebot script loads or fails", () => {
    const script = document.createElement("script");
    script.id = "Cookiebot";
    document.body.append(script);
    const listener = vi.fn();

    const unsubscribe = subscribeToConsentManagerStatus(listener);
    script.dispatchEvent(new Event("load"));
    script.dispatchEvent(new Event("error"));
    window.dispatchEvent(new Event("load"));
    unsubscribe();
    script.dispatchEvent(new Event("load"));

    expect(listener).toHaveBeenCalledTimes(3);
    script.remove();
  });
});
