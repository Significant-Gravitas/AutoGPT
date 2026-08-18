import { beforeEach, describe, expect, it } from "vitest";
import { Key } from "../storage/local-storage";
import { consent, ConsentPreferences } from "./cookies";

const COOKIE_NAME = "agpt_analytics_consent";

function preferences(analytics: boolean): ConsentPreferences {
  return {
    hasConsented: true,
    timestamp: Date.now(),
    analytics,
    monitoring: false,
  };
}

describe("analytics consent cookie", () => {
  beforeEach(() => {
    localStorage.clear();
    document.cookie = `${COOKIE_NAME}=; Path=/; Max-Age=0`;
  });

  it("mirrors granted analytics consent for server-side tracking", () => {
    consent.save(preferences(true));

    expect(document.cookie).toContain(`${COOKIE_NAME}=granted`);
  });

  it("removes server-readable consent when analytics is denied", () => {
    document.cookie = `${COOKIE_NAME}=granted; Path=/`;

    consent.save(preferences(false));

    expect(document.cookie).not.toContain(`${COOKIE_NAME}=granted`);
  });

  it("migrates existing localStorage consent when it is loaded", () => {
    localStorage.setItem(Key.COOKIE_CONSENT, JSON.stringify(preferences(true)));

    consent.load();

    expect(document.cookie).toContain(`${COOKIE_NAME}=granted`);
  });

  it("revokes a stale server cookie when no stored consent exists", () => {
    document.cookie = `${COOKIE_NAME}=granted; Path=/`;

    consent.load();

    expect(document.cookie).not.toContain(`${COOKIE_NAME}=granted`);
  });

  it("revokes a stale server cookie when stored consent is invalid", () => {
    document.cookie = `${COOKIE_NAME}=granted; Path=/`;
    localStorage.setItem(Key.COOKIE_CONSENT, "invalid-json");

    consent.load();

    expect(document.cookie).not.toContain(`${COOKIE_NAME}=granted`);
  });

  it("removes the server-readable consent cookie when consent is cleared", () => {
    consent.save(preferences(true));

    consent.clear();

    expect(document.cookie).not.toContain(`${COOKIE_NAME}=granted`);
  });
});
