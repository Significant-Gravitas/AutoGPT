import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const posthog = vi.hoisted(() => ({
  identify: vi.fn(),
  setPersonPropertiesForFlags: vi.fn(),
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
  environment: { isPostHogEnabled: () => true },
}));

vi.mock("@/services/feature-flags/flag-backend", () => ({
  usesPostHog: () => flags.usesPostHog,
}));

describe("PostHogUserTracker", () => {
  beforeEach(() => {
    posthog.identify.mockClear();
    posthog.setPersonPropertiesForFlags.mockClear();
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
      email_domain: "example.com",
      role: "authenticated",
      created_at: "2026-05-08T12:00:00Z",
    });
  });

  it("leaves flag properties alone while LaunchDarkly answers flags", async () => {
    flags.usesPostHog = false;
    const { PostHogUserTracker } = await import("../posthog-provider");

    render(<PostHogUserTracker />);

    expect(posthog.setPersonPropertiesForFlags).not.toHaveBeenCalled();
  });
});
