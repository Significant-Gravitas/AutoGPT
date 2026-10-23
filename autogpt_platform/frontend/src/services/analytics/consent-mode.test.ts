import { afterEach, describe, expect, it, vi } from "vitest";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import {
  buildConsentDefaultsScript,
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

function update(analytics: boolean, advertising: boolean, preferences = false) {
  const ads = advertising ? "granted" : "denied";
  const prefs = preferences ? "granted" : "denied";
  return [
    "consent",
    "update",
    {
      analytics_storage: analytics ? "granted" : "denied",
      ad_storage: ads,
      ad_user_data: ads,
      ad_personalization: ads,
      functionality_storage: prefs,
      personalization_storage: prefs,
    },
  ];
}

function runDefaultsScript(): unknown[][] {
  new Function(buildConsentDefaultsScript())();
  return ((window as DataLayerWindow).dataLayer ?? []).map((entry) =>
    Array.from(entry),
  );
}

const DEFAULTS = {
  ad_storage: "denied",
  ad_user_data: "denied",
  ad_personalization: "denied",
  analytics_storage: "denied",
  functionality_storage: "denied",
  personalization_storage: "denied",
  security_storage: "granted",
  wait_for_update: 500,
};

// Every Consent Mode v2 signal; Google treats one without a default as not
// denied.
const CONSENT_SIGNALS = [
  "ad_storage",
  "ad_user_data",
  "ad_personalization",
  "analytics_storage",
  "functionality_storage",
  "personalization_storage",
  "security_storage",
];

describe("buildConsentDefaultsScript", () => {
  afterEach(() => {
    delete (window as DataLayerWindow).dataLayer;
    delete window.gtag;
  });

  it("denies every signal that needs consent for every visitor and passes click IDs through URLs", () => {
    expect(runDefaultsScript()).toEqual([
      ["consent", "default", DEFAULTS],
      ["set", "url_passthrough", true],
    ]);
  });

  it("sets a default for every Consent Mode signal, granting only security storage", () => {
    const [[, , defaults]] = runDefaultsScript() as [
      [string, string, Record<string, unknown>],
    ];

    expect(Object.keys(defaults)).toEqual(
      expect.arrayContaining(CONSENT_SIGNALS),
    );
    const granted = CONSENT_SIGNALS.filter(
      (name) => defaults[name] === "granted",
    );
    expect(granted).toEqual(["security_storage"]);
  });

  it("sets no region-specific default", () => {
    // The vendored gtag.js resolves every visitor to the location baked into
    // it, so a region-specific default would apply to everyone.
    expect(buildConsentDefaultsScript()).not.toContain("region");
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

  it("grants consent once the visitor allows it", () => {
    runDefaultsScript();
    configureCookiebot();
    installCookiebot();

    const unfollow = followConsentForGoogleTag();
    expect(consentUpdates()).toEqual([]);

    answerCookiebot({ statistics: true, marketing: true });

    expect(dataLayerEntries()).toEqual([
      ["consent", "default", DEFAULTS],
      ["set", "url_passthrough", true],
      [
        "consent",
        "update",
        {
          analytics_storage: "granted",
          ad_storage: "granted",
          ad_user_data: "granted",
          ad_personalization: "granted",
          functionality_storage: "denied",
          personalization_storage: "denied",
        },
      ],
    ]);
    unfollow();
  });

  it("sends preferences consent as the functionality and personalization signals", () => {
    configureCookiebot();
    installCookiebot();
    const unfollow = followConsentForGoogleTag();

    answerCookiebot({ preferences: true });
    answerCookiebot({ statistics: true });

    expect(consentUpdates()).toEqual([
      update(false, false, true),
      update(true, false, false),
    ]);
    unfollow();
  });

  it("never updates security storage, which stays granted", () => {
    configureCookiebot();
    installCookiebot({ statistics: true, marketing: true, preferences: true });
    const unfollow = followConsentForGoogleTag();

    answerCookiebot({});

    expect(consentUpdates()).toHaveLength(2);
    consentUpdates().forEach(([, , signals]) =>
      expect(signals).not.toHaveProperty("security_storage"),
    );
    unfollow();
  });

  it("keeps every signal denied when the visitor declines", () => {
    runDefaultsScript();
    configureCookiebot();
    installCookiebot();

    const unfollow = followConsentForGoogleTag();
    answerCookiebot({});

    expect(consentUpdates()).toEqual([update(false, false)]);
    unfollow();
  });

  it("leaves the denied defaults alone until the visitor answers", () => {
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
