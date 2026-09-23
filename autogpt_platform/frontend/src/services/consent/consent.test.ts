import { afterEach, describe, expect, it, vi } from "vitest";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import {
  getConsent,
  hasConsentFor,
  isConsentManagerConfigured,
  NO_CONSENT,
  openConsentSettings,
  subscribeToConsent,
  toConsentState,
} from "./consent";
import { readConsentFromCookies } from "./consent-server";

const STATISTICS_ONLY =
  "{stamp:'abc',necessary:true,preferences:false,statistics:true,marketing:false,method:'explicit',ver:1,utc:1,region:'de'}";

function storeCookie(value: string) {
  document.cookie = `CookieConsent=${encodeURIComponent(value)}; Path=/`;
}

afterEach(() => {
  removeCookiebot();
  vi.unstubAllEnvs();
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
    const store = { get: () => ({ value: "-1" }) };

    expect(readConsentFromCookies(store)).toEqual(NO_CONSENT);
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
    const store = {
      get: (name: string) =>
        name === "CookieConsent"
          ? { value: encodeURIComponent(STATISTICS_ONLY) }
          : undefined,
    };

    expect(readConsentFromCookies(store)).toEqual({
      analytics: true,
      monitoring: true,
      advertising: false,
    });
    expect(readConsentFromCookies({ get: () => undefined })).toEqual(
      NO_CONSENT,
    );
  });
});
