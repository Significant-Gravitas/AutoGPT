import { afterEach, describe, expect, it, vi } from "vitest";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import {
  buildConsentDefaultsScript,
  CONSENT_DENIED_BY_DEFAULT_REGIONS,
  followConsentForGoogleTag,
} from "./consent-mode";

type DataLayerWindow = Window & { dataLayer?: IArguments[] };

function dataLayerEntries(): unknown[][] {
  return ((window as DataLayerWindow).dataLayer ?? []).map((entry) =>
    Array.from(entry),
  );
}

function consentUpdates() {
  return dataLayerEntries().filter(
    ([command, action]) => command === "consent" && action === "update",
  );
}

function update(analytics: boolean, advertising: boolean) {
  const ads = advertising ? "granted" : "denied";
  return [
    "consent",
    "update",
    {
      analytics_storage: analytics ? "granted" : "denied",
      ad_storage: ads,
      ad_user_data: ads,
      ad_personalization: ads,
    },
  ];
}

function runDefaultsScript(): unknown[][] {
  new Function(buildConsentDefaultsScript())();
  return ((window as DataLayerWindow).dataLayer ?? []).map((entry) =>
    Array.from(entry),
  );
}

describe("buildConsentDefaultsScript", () => {
  afterEach(() => {
    delete (window as DataLayerWindow).dataLayer;
    delete window.gtag;
  });

  it("grants by default, denies in the EEA, UK and Switzerland, and passes click IDs through URLs", () => {
    expect(runDefaultsScript()).toEqual([
      [
        "consent",
        "default",
        {
          ad_storage: "granted",
          ad_user_data: "granted",
          ad_personalization: "granted",
          analytics_storage: "granted",
        },
      ],
      [
        "consent",
        "default",
        {
          ad_storage: "denied",
          ad_user_data: "denied",
          ad_personalization: "denied",
          analytics_storage: "denied",
          region: CONSENT_DENIED_BY_DEFAULT_REGIONS,
          wait_for_update: 500,
        },
      ],
      ["set", "url_passthrough", true],
    ]);
    expect(CONSENT_DENIED_BY_DEFAULT_REGIONS).toEqual(
      expect.arrayContaining(["DE", "FR", "ES", "GB", "CH", "NO", "IS", "LI"]),
    );
  });

  it("only sets defaults; updates come from the visitor's answer", () => {
    expect(buildConsentDefaultsScript()).not.toContain("'update'");
  });

  it("queues real arguments objects without defining window.gtag", () => {
    runDefaultsScript();

    const [first] = (window as DataLayerWindow).dataLayer ?? [];
    expect(Object.prototype.toString.call(first)).toBe("[object Arguments]");
    expect(window.gtag).toBeUndefined();
  });

  it("keeps entries already in the dataLayer", () => {
    (window as DataLayerWindow).dataLayer = [];
    const existing = (window as DataLayerWindow).dataLayer;

    runDefaultsScript();

    expect((window as DataLayerWindow).dataLayer).toBe(existing);
  });
});

describe("followConsentForGoogleTag", () => {
  afterEach(() => {
    delete (window as DataLayerWindow).dataLayer;
    removeCookiebot();
    vi.unstubAllEnvs();
  });

  it("sends a stored answer straight away, as real arguments objects", () => {
    configureCookiebot();
    installCookiebot({ statistics: true });

    const unfollow = followConsentForGoogleTag();

    expect(consentUpdates()).toEqual([update(true, false)]);
    const [entry] = (window as DataLayerWindow).dataLayer ?? [];
    expect(Object.prototype.toString.call(entry)).toBe("[object Arguments]");
    unfollow();
  });

  it("leaves the region defaults alone until the visitor answers", () => {
    configureCookiebot();
    installCookiebot();

    const unfollow = followConsentForGoogleTag();
    expect(consentUpdates()).toEqual([]);

    answerCookiebot({ statistics: true, marketing: true });
    expect(consentUpdates()).toEqual([update(true, true)]);
    unfollow();
  });

  it("sends a denial, and every later change, once per change", () => {
    configureCookiebot();
    installCookiebot({ statistics: true, marketing: true });
    const unfollow = followConsentForGoogleTag();

    answerCookiebot({});
    window.dispatchEvent(new Event("CookiebotOnLoad"));
    answerCookiebot({ marketing: true });

    expect(consentUpdates()).toEqual([
      update(true, true),
      update(false, false),
      update(false, true),
    ]);
    unfollow();
  });

  it("denies an answer Cookiebot has since withdrawn", () => {
    configureCookiebot();
    installCookiebot({ statistics: true });
    const unfollow = followConsentForGoogleTag();

    const cookiebot = window.Cookiebot;
    if (!cookiebot) throw new Error("Cookiebot missing");
    cookiebot.hasResponse = false;
    window.dispatchEvent(new Event("CookiebotOnLoad"));

    expect(consentUpdates()).toEqual([
      update(true, false),
      update(false, false),
    ]);
    unfollow();
  });

  it("stops after unsubscribe", () => {
    configureCookiebot();
    installCookiebot();
    followConsentForGoogleTag()();

    answerCookiebot({ statistics: true });

    expect(consentUpdates()).toEqual([]);
  });

  it("sends nothing without a Cookiebot domain group", () => {
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");
    installCookiebot({ statistics: true });

    const unfollow = followConsentForGoogleTag();
    answerCookiebot({ statistics: true, marketing: true });

    expect(consentUpdates()).toEqual([]);
    unfollow();
  });
});
