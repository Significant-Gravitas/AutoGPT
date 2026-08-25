import { describe, expect, it } from "vitest";
import type { ConsentPreferences } from "@/services/consent/cookies";
import { resolveAnalyticsLoading } from "./loading-policy";

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

const production = {
  host: "platform.agpt.co",
  pathname: "/marketplace",
  isLocal: false,
};

describe("resolveAnalyticsLoading", () => {
  it("loads the Google tag on production once preferences are known, even unanswered", () => {
    const result = resolveAnalyticsLoading({
      ...production,
      preferences: preferences({ hasConsented: false }),
    });

    expect(result.googleTag).toBe(true);
    expect(result.dataFast).toBe(false);
  });

  it("waits for the stored preferences before loading anything", () => {
    const result = resolveAnalyticsLoading({
      ...production,
      preferences: null,
    });

    expect(result).toEqual({ googleTag: false, dataFast: false });
  });

  it("keeps the Google tag off non-production cloud domains", () => {
    const result = resolveAnalyticsLoading({
      ...production,
      host: "dev-builder.agpt.co",
      preferences: preferences({ analytics: true, advertising: true }),
    });

    expect(result.googleTag).toBe(false);
  });

  it("loads the Google tag locally only with analytics consent", () => {
    const local = { host: "localhost:3000", pathname: "/", isLocal: true };

    expect(
      resolveAnalyticsLoading({ ...local, preferences: preferences() })
        .googleTag,
    ).toBe(false);
    expect(
      resolveAnalyticsLoading({
        ...local,
        preferences: preferences({ analytics: true }),
      }).googleTag,
    ).toBe(true);
  });

  it("loads DataFast on production with analytics consent", () => {
    const result = resolveAnalyticsLoading({
      ...production,
      preferences: preferences({ analytics: true }),
    });

    expect(result.dataFast).toBe(true);
  });

  it("loads DataFast on the public tour without consent", () => {
    const result = resolveAnalyticsLoading({
      ...production,
      pathname: "/tour/chat",
      preferences: preferences({ hasConsented: false }),
    });

    expect(result.dataFast).toBe(true);
  });

  it("does not treat /tourism as the tour", () => {
    const result = resolveAnalyticsLoading({
      ...production,
      pathname: "/tourism",
      preferences: preferences({ hasConsented: false }),
    });

    expect(result.dataFast).toBe(false);
  });
});
