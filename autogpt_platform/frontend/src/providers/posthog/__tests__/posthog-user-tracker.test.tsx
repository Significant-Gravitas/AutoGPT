import {
  installCookiebot,
  removeCookiebot,
  TEST_COOKIEBOT_CBID,
} from "@/tests/integrations/cookiebot";
import { render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const posthog = vi.hoisted(() => ({
  identify: vi.fn(),
  setPersonPropertiesForFlags: vi.fn(),
}));

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

describe("PostHogUserTracker", () => {
  beforeEach(() => {
    posthog.identify.mockClear();
    posthog.setPersonPropertiesForFlags.mockClear();
    installCookiebot({ statistics: true });
  });

  afterEach(() => {
    removeCookiebot();
  });

  it("identifies with the analytics properties only", async () => {
    const { PostHogUserTracker } = await import("../posthog-provider");

    render(<PostHogUserTracker />);

    expect(posthog.identify).toHaveBeenCalledWith("user-1", {
      email: "user@example.com",
      name: "Ada",
    });
    // Flags are read on the server, so the browser hands PostHog no
    // targeting attributes.
    expect(posthog.setPersonPropertiesForFlags).not.toHaveBeenCalled();
  });

  it("hands PostHog neither identity nor flag properties without analytics consent", async () => {
    removeCookiebot();
    installCookiebot({ statistics: false });
    const { PostHogUserTracker } = await import("../posthog-provider");

    render(<PostHogUserTracker />);

    expect(posthog.identify).not.toHaveBeenCalled();
    expect(posthog.setPersonPropertiesForFlags).not.toHaveBeenCalled();
  });
});
