import { describe, expect, it } from "vitest";
import { NO_CONSENT, type ConsentState } from "@/services/consent/consent";
import { resolveAnalyticsLoading } from "./loading-policy";

function consent(overrides: Partial<ConsentState> = {}): ConsentState {
  return { ...NO_CONSENT, ...overrides };
}

const production = {
  host: "platform.agpt.co",
  pathname: "/marketplace",
  isLocal: false,
  isConsentManaged: true,
};

describe("resolveAnalyticsLoading", () => {
  it("loads the Google tag on production before the visitor answers", () => {
    const result = resolveAnalyticsLoading({
      ...production,
      consent: consent(),
    });

    expect(result.googleTag).toBe(true);
    expect(result.dataFast).toBe(false);
  });

  it("keeps the Google tag off production when no banner is configured", () => {
    const result = resolveAnalyticsLoading({
      ...production,
      isConsentManaged: false,
      consent: consent(),
    });

    expect(result.googleTag).toBe(false);
  });

  it("keeps the Google tag off non-production cloud domains", () => {
    const result = resolveAnalyticsLoading({
      ...production,
      host: "dev-builder.agpt.co",
      consent: consent({ analytics: true, advertising: true }),
    });

    expect(result.googleTag).toBe(false);
  });

  it("rejects hostnames that merely contain the production host", () => {
    const consented = consent({ analytics: true, advertising: true });

    for (const host of [
      "platform.agpt.co.example.com",
      "notplatform.agpt.co",
      "platform.agpt.co.evil.co",
      "agpt.co",
    ]) {
      const result = resolveAnalyticsLoading({
        ...production,
        host,
        consent: consented,
      });

      expect({ host, ...result }).toEqual({
        host,
        googleTag: false,
        dataFast: false,
      });
    }
  });

  it("accepts the production host with a port, odd casing or a trailing dot", () => {
    const consented = consent({ analytics: true, advertising: true });

    for (const host of [
      "platform.agpt.co:443",
      "Platform.AGPT.co",
      "platform.agpt.co.",
    ]) {
      const result = resolveAnalyticsLoading({
        ...production,
        host,
        consent: consented,
      });

      expect({ host, ...result }).toEqual({
        host,
        googleTag: true,
        dataFast: true,
      });
    }
  });

  it("loads the Google tag locally only with analytics consent", () => {
    const local = {
      host: "localhost:3000",
      pathname: "/",
      isLocal: true,
      isConsentManaged: true,
    };

    expect(
      resolveAnalyticsLoading({ ...local, consent: consent() }).googleTag,
    ).toBe(false);
    expect(
      resolveAnalyticsLoading({
        ...local,
        consent: consent({ analytics: true }),
      }).googleTag,
    ).toBe(true);
  });

  it("loads DataFast on production with analytics consent", () => {
    const result = resolveAnalyticsLoading({
      ...production,
      consent: consent({ analytics: true }),
    });

    expect(result.dataFast).toBe(true);
  });

  it("loads DataFast on the public tour without consent", () => {
    const result = resolveAnalyticsLoading({
      ...production,
      pathname: "/tour/chat",
      consent: consent(),
    });

    expect(result.dataFast).toBe(true);
  });

  it("keeps DataFast off the tour when no banner is configured", () => {
    const result = resolveAnalyticsLoading({
      ...production,
      pathname: "/tour",
      isConsentManaged: false,
      consent: consent(),
    });

    expect(result).toEqual({ googleTag: false, dataFast: false });
  });

  it("does not treat /tourism as the tour", () => {
    const result = resolveAnalyticsLoading({
      ...production,
      pathname: "/tourism",
      consent: consent(),
    });

    expect(result.dataFast).toBe(false);
  });
});
