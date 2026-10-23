import { DEFAULT_SEARCH_TERMS } from "@/app/(platform)/marketplace/components/HeroSection/helpers";
import { describe, expect, it } from "vitest";
import { Flag, resolveFlagValue } from "../use-get-flag";

// PostHog answers a flag with no payload as a bare boolean, which for a
// JSON-valued flag would otherwise reach consumers that call `.map` on it.
describe("resolveFlagValue", () => {
  it("falls back when an array flag is answered with a boolean", () => {
    expect(resolveFlagValue(Flag.MARKETPLACE_SEARCH_TERMS, true)).toEqual(
      DEFAULT_SEARCH_TERMS,
    );
  });

  it("falls back when an array flag is answered with an object", () => {
    expect(resolveFlagValue(Flag.MARKETPLACE_SEARCH_TERMS, { a: 1 })).toEqual(
      DEFAULT_SEARCH_TERMS,
    );
  });

  it("keeps a real array value for an array flag", () => {
    expect(resolveFlagValue(Flag.MARKETPLACE_SEARCH_TERMS, ["a", "b"])).toEqual(
      ["a", "b"],
    );
  });

  it("falls back when an object flag is answered with a boolean", () => {
    expect(resolveFlagValue(Flag.COPILOT_BOT_PLATFORMS, true)).toEqual({});
  });

  it("falls back when an object flag is answered with an array", () => {
    expect(resolveFlagValue(Flag.COPILOT_BOT_PLATFORMS, [])).toEqual({});
  });

  it("keeps a real object value for an object flag", () => {
    expect(
      resolveFlagValue(Flag.COPILOT_BOT_PLATFORMS, { discord: true }),
    ).toEqual({ discord: true });
  });

  it("keeps a boolean for a boolean flag, both ways", () => {
    expect(resolveFlagValue(Flag.ENABLE_PLATFORM_PAYMENT, true)).toBe(true);
    expect(resolveFlagValue(Flag.ENABLE_PLATFORM_PAYMENT, false)).toBe(false);
  });

  it("falls back when a boolean flag is answered with a string", () => {
    expect(resolveFlagValue(Flag.ENABLE_PLATFORM_PAYMENT, "on")).toBe(false);
  });

  it("falls back on null and undefined", () => {
    expect(resolveFlagValue(Flag.ENABLE_PLATFORM_PAYMENT, null)).toBe(false);
    expect(resolveFlagValue(Flag.ENABLE_PLATFORM_PAYMENT, undefined)).toBe(
      false,
    );
  });
});
