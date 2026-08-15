import { afterEach, describe, expect, test } from "vitest";
import {
  isNamingMomentEligible,
  peekNamingMomentDismissed,
  setNamingMomentDismissed,
} from "./helpers";

const base = {
  isExpertsEnabled: true,
  isFlagReady: true,
  isLoaded: true,
  hasExperts: false,
  hasSessions: true,
  isDismissed: false,
};

describe("isNamingMomentEligible", () => {
  test("existing user with sessions and no experts is eligible", () => {
    expect(isNamingMomentEligible(base)).toBe(true);
  });

  test("fresh user with no sessions is not eligible", () => {
    expect(isNamingMomentEligible({ ...base, hasSessions: false })).toBe(false);
  });

  test("user who already has an expert is not eligible", () => {
    expect(isNamingMomentEligible({ ...base, hasExperts: true })).toBe(false);
  });

  test("dismissed user is not eligible", () => {
    expect(isNamingMomentEligible({ ...base, isDismissed: true })).toBe(false);
  });

  test("flag off is not eligible", () => {
    expect(isNamingMomentEligible({ ...base, isExpertsEnabled: false })).toBe(
      false,
    );
  });

  test("flag not ready is not eligible", () => {
    expect(isNamingMomentEligible({ ...base, isFlagReady: false })).toBe(false);
  });

  test("queries not settled is not eligible", () => {
    expect(isNamingMomentEligible({ ...base, isLoaded: false })).toBe(false);
  });
});

describe("naming moment dismissal persistence", () => {
  afterEach(() => {
    window.localStorage.clear();
  });

  test("persists dismissal keyed by user id", () => {
    expect(peekNamingMomentDismissed("user-1")).toBe(false);
    setNamingMomentDismissed("user-1");
    expect(peekNamingMomentDismissed("user-1")).toBe(true);
  });

  test("a different user on the same browser starts clean", () => {
    setNamingMomentDismissed("user-1");
    expect(peekNamingMomentDismissed("user-2")).toBe(false);
  });

  test("a missing user id never persists or reads as dismissed", () => {
    setNamingMomentDismissed(null);
    expect(peekNamingMomentDismissed(null)).toBe(false);
  });

  test("user B dismissing does not clobber user A's dismissal", () => {
    setNamingMomentDismissed("user-a");
    setNamingMomentDismissed("user-b");
    expect(peekNamingMomentDismissed("user-a")).toBe(true);
    expect(peekNamingMomentDismissed("user-b")).toBe(true);
  });

  test("a dismissal stored under the legacy shared key still counts", () => {
    window.localStorage.setItem("autogpt:naming-moment-dismissed", "user-1");
    expect(peekNamingMomentDismissed("user-1")).toBe(true);
    expect(peekNamingMomentDismissed("user-2")).toBe(false);
  });
});
