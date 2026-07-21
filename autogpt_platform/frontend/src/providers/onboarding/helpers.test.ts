import { describe, expect, test } from "vitest";
import {
  decideOnboardingRedirect,
  shouldRedirectFromOnboarding,
} from "./helpers";

describe("shouldRedirectFromOnboarding", () => {
  test("returns true once VISIT_COPILOT is completed", () => {
    expect(shouldRedirectFromOnboarding(["VISIT_COPILOT"], "/onboarding")).toBe(
      true,
    );
  });

  test("returns false when VISIT_COPILOT is missing", () => {
    expect(shouldRedirectFromOnboarding([], "/onboarding")).toBe(false);
  });

  test("does not redirect away from /onboarding/reset", () => {
    expect(
      shouldRedirectFromOnboarding(["VISIT_COPILOT"], "/onboarding/reset"),
    ).toBe(false);
  });
});

describe("decideOnboardingRedirect", () => {
  // Defaults — most tests vary one bit at a time.
  const base = {
    isCompleted: false,
    isOnOnboardingRoute: false,
    isOnAuthRoute: false,
    hasPendingAuthDeepLink: false,
  };

  test("pending deep link defers to the auth page (returns null)", () => {
    // Both completed and incomplete users must yield — the auth page's
    // `router.replace(nextUrl)` is in flight and we must not clobber it.
    expect(
      decideOnboardingRedirect({
        ...base,
        isOnAuthRoute: true,
        hasPendingAuthDeepLink: true,
      }),
    ).toBeNull();
    expect(
      decideOnboardingRedirect({
        ...base,
        isCompleted: true,
        isOnAuthRoute: true,
        hasPendingAuthDeepLink: true,
      }),
    ).toBeNull();
  });

  test("incomplete user on any non-onboarding route is sent to /onboarding", () => {
    // /signup, /login, /copilot, /library — all should funnel to /onboarding.
    expect(decideOnboardingRedirect(base)).toBe("/onboarding");
    expect(decideOnboardingRedirect({ ...base, isOnAuthRoute: true })).toBe(
      "/onboarding",
    );
  });

  test("incomplete user already on /onboarding stays put", () => {
    expect(
      decideOnboardingRedirect({ ...base, isOnOnboardingRoute: true }),
    ).toBeNull();
  });

  test("completed user on /onboarding is sent to /copilot", () => {
    expect(
      decideOnboardingRedirect({
        ...base,
        isCompleted: true,
        isOnOnboardingRoute: true,
      }),
    ).toBe("/copilot");
  });

  test("completed user on /signup or /login is sent to /copilot", () => {
    expect(
      decideOnboardingRedirect({
        ...base,
        isCompleted: true,
        isOnAuthRoute: true,
      }),
    ).toBe("/copilot");
  });

  test("completed user already on the product is left alone", () => {
    // E.g., /copilot, /library — no redirect needed.
    expect(decideOnboardingRedirect({ ...base, isCompleted: true })).toBeNull();
  });
});
