import {
  installCookiebot,
  removeCookiebot,
  TEST_COOKIEBOT_CBID,
} from "@/tests/integrations/cookiebot";
import { render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const flagsOnly = vi.hoisted(() => ({
  identify: vi.fn(),
  setPersonPropertiesForFlags: vi.fn(),
}));
const posthog = vi.hoisted(() => ({
  identify: vi.fn(),
  setPersonPropertiesForFlags: vi.fn(),
  init: vi.fn(() => flagsOnly),
}));
const flags = vi.hoisted(() => ({ usesPostHog: false }));

vi.mock("posthog-js", () => ({ default: posthog }));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    isUserLoading: false,
    user: {
      id: "user-1",
      email: "user@example.com",
      role: "authenticated",
      created_at: "2026-05-08T12:00:00Z",
      user_metadata: { name: "Ada" },
    },
  }),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    isPostHogEnabled: () => true,
    getCookiebotCBID: () => TEST_COOKIEBOT_CBID,
  },
}));

vi.mock("@/services/feature-flags/flag-backend", () => ({
  usesPostHog: () => flags.usesPostHog,
}));

describe("PostHogUserTracker", () => {
  beforeEach(() => {
    vi.resetModules();
    posthog.identify.mockClear();
    posthog.setPersonPropertiesForFlags.mockClear();
    flagsOnly.identify.mockClear();
    flagsOnly.setPersonPropertiesForFlags.mockClear();
    installCookiebot({ statistics: true });
  });

  afterEach(() => {
    removeCookiebot();
  });

  it.each([false, true])(
    "identifies with the analytics properties only (PostHog flags: %s)",
    async (usesPostHog) => {
      flags.usesPostHog = usesPostHog;
      const { PostHogUserTracker } = await import("../posthog-provider");

      render(<PostHogUserTracker />);

      expect(posthog.identify).toHaveBeenCalledWith("user-1", {
        email: "user@example.com",
        name: "Ada",
      });
    },
  );

  it("hands the targeting attributes to flag evaluation alone", async () => {
    flags.usesPostHog = true;
    const { PostHogUserTracker } = await import("../posthog-provider");

    render(<PostHogUserTracker />);

    expect(posthog.setPersonPropertiesForFlags).toHaveBeenCalledWith({
      email: "user@example.com",
      email_domain: "example.com",
      role: "authenticated",
      created_at: "2026-05-08T12:00:00Z",
    });
  });

  it("identifies a flags-only instance too, and hands it the flag properties", async () => {
    flags.usesPostHog = true;
    const { initFlagsOnlyClient } = await import("../posthog-flags-client");
    initFlagsOnlyClient({
      flagsKey: "phc_prod",
      analyticsKey: "phc_dev",
      hosts: {},
      anonymousID: null,
    });
    const { PostHogUserTracker } = await import("../posthog-provider");

    render(<PostHogUserTracker />);

    expect(flagsOnly.identify).toHaveBeenCalledWith("user-1");
    expect(flagsOnly.setPersonPropertiesForFlags).toHaveBeenCalledWith(
      expect.objectContaining({ email: "user@example.com" }),
    );
    expect(posthog.setPersonPropertiesForFlags).not.toHaveBeenCalled();
    expect(posthog.identify).toHaveBeenCalledWith("user-1", {
      email: "user@example.com",
      name: "Ada",
    });
  });

  it("leaves flag properties alone while LaunchDarkly answers flags", async () => {
    flags.usesPostHog = false;
    const { PostHogUserTracker } = await import("../posthog-provider");

    render(<PostHogUserTracker />);

    expect(posthog.setPersonPropertiesForFlags).not.toHaveBeenCalled();
  });

  it("hands PostHog neither identity nor flag properties without analytics consent", async () => {
    flags.usesPostHog = true;
    removeCookiebot();
    installCookiebot({ statistics: false });
    const { PostHogUserTracker } = await import("../posthog-provider");

    render(<PostHogUserTracker />);

    expect(posthog.identify).not.toHaveBeenCalled();
    expect(posthog.setPersonPropertiesForFlags).not.toHaveBeenCalled();
  });
});
