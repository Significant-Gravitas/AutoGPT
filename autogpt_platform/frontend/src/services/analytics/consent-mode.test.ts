import { describe, expect, it } from "vitest";
import type { ConsentPreferences } from "@/services/consent/cookies";
import {
  buildConsentModeScript,
  CONSENT_DENIED_BY_DEFAULT_REGIONS,
} from "./consent-mode";

function preferences(
  overrides: Partial<ConsentPreferences> = {},
): ConsentPreferences {
  return {
    hasConsented: true,
    timestamp: 1,
    analytics: false,
    monitoring: false,
    advertising: false,
    ...overrides,
  };
}

describe("buildConsentModeScript", () => {
  it("grants by default, denies in the EEA, UK and Switzerland, and passes click IDs through URLs", () => {
    const script = buildConsentModeScript(null);

    expect(script).toContain(
      `gtag('consent','default',{"ad_storage":"granted","ad_user_data":"granted","ad_personalization":"granted","analytics_storage":"granted"});`,
    );
    expect(script).toContain(
      `gtag('consent','default',{"ad_storage":"denied","ad_user_data":"denied","ad_personalization":"denied","analytics_storage":"denied","region":${JSON.stringify(CONSENT_DENIED_BY_DEFAULT_REGIONS)}});`,
    );
    expect(CONSENT_DENIED_BY_DEFAULT_REGIONS).toEqual(
      expect.arrayContaining(["DE", "FR", "ES", "GB", "CH", "NO", "IS", "LI"]),
    );
    expect(script).toContain(`gtag('set','url_passthrough',true);`);
    expect(script).not.toContain("'update'");
  });

  it("sends no update while the visitor has not answered the banner", () => {
    const script = buildConsentModeScript(
      preferences({ hasConsented: false, analytics: true }),
    );

    expect(script).not.toContain("'update'");
  });

  it("updates every signal from the stored answer", () => {
    const script = buildConsentModeScript(
      preferences({ analytics: true, advertising: false }),
    );

    expect(script).toContain(
      `gtag('consent','update',{"analytics_storage":"granted","ad_storage":"denied","ad_user_data":"denied","ad_personalization":"denied"});`,
    );
  });

  it("grants the advertising signals once advertising is accepted", () => {
    const script = buildConsentModeScript(
      preferences({ analytics: false, advertising: true }),
    );

    expect(script).toContain(
      `gtag('consent','update',{"analytics_storage":"denied","ad_storage":"granted","ad_user_data":"granted","ad_personalization":"granted"});`,
    );
  });
});
