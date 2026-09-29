import { afterEach, describe, expect, it, vi } from "vitest";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import {
  buildConsentDefaultsScript,
  CONSENT_GRANTED_BY_DEFAULT_REGIONS,
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

type ConsentDefault = Record<string, unknown> & { region?: string[] };

// Google's rule: the default whose region lists the visitor's country wins,
// and the one without a region covers everyone else.
function defaultsFor(country: string) {
  const defaults = dataLayerEntries()
    .filter(
      ([command, action]) => command === "consent" && action === "default",
    )
    .map(([, , params]) => params as ConsentDefault);
  const regional = defaults.find((params) => params.region?.includes(country));
  const { region: _region, ...signals } =
    regional ?? defaults.find((params) => !params.region) ?? {};
  return signals;
}

const ALL_DENIED = {
  ad_storage: "denied",
  ad_user_data: "denied",
  ad_personalization: "denied",
  analytics_storage: "denied",
  wait_for_update: 500,
};

const ALL_GRANTED = {
  ad_storage: "granted",
  ad_user_data: "granted",
  ad_personalization: "granted",
  analytics_storage: "granted",
  wait_for_update: 500,
};

describe("buildConsentDefaultsScript", () => {
  afterEach(() => {
    delete (window as DataLayerWindow).dataLayer;
    delete window.gtag;
  });

  it("grants only in the US, denies everywhere else, and passes click IDs through URLs", () => {
    expect(runDefaultsScript()).toEqual([
      [
        "consent",
        "default",
        { ...ALL_GRANTED, region: CONSENT_GRANTED_BY_DEFAULT_REGIONS },
      ],
      ["consent", "default", ALL_DENIED],
      ["set", "url_passthrough", true],
    ]);
    expect(CONSENT_GRANTED_BY_DEFAULT_REGIONS).toEqual(["US"]);
  });

  it.each(["BR", "CA", "IN", "DE", "GB", "CH", "AU"])(
    "starts a visitor in %s with every signal denied",
    (country) => {
      runDefaultsScript();

      expect(defaultsFor(country)).toEqual(ALL_DENIED);
    },
  );

  it("starts a US visitor with every signal granted", () => {
    runDefaultsScript();

    expect(defaultsFor("US")).toEqual(ALL_GRANTED);
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

  it("follows the defaults with the visitor's answer", () => {
    runDefaultsScript();
    configureCookiebot();
    installCookiebot();

    const unfollow = followConsentForGoogleTag();
    answerCookiebot({ statistics: true, marketing: true });

    expect(dataLayerEntries()).toEqual([
      ["consent", "default", expect.objectContaining({ region: ["US"] })],
      ["consent", "default", ALL_DENIED],
      ["set", "url_passthrough", true],
      update(true, true),
    ]);
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
