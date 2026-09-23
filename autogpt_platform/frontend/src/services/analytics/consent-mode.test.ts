import { afterEach, describe, expect, it } from "vitest";
import {
  buildConsentDefaultsScript,
  CONSENT_DENIED_BY_DEFAULT_REGIONS,
} from "./consent-mode";

type DataLayerWindow = Window & { dataLayer?: IArguments[] };

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

  it("leaves every consent update to Cookiebot", () => {
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
