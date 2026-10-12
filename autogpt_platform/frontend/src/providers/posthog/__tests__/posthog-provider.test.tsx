import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const posthog = vi.hoisted(() => ({
  init: vi.fn(),
  register: vi.fn(),
}));
const credentials = vi.hoisted(() => ({ key: "phc_test" as string }));

vi.mock("posthog-js", () => ({ default: posthog }));

vi.mock("@posthog/react", () => ({
  PostHogProvider: ({ children }: { children: React.ReactNode }) => children,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isUserLoading: true, user: null }),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    isPostHogEnabled: () => true,
    getPostHogCredentials: () => ({
      key: credentials.key,
      host: "https://eu.i.posthog.com",
    }),
  },
}));

vi.mock("@/services/analytics/posthog-base-properties", () => ({
  getPostHogBaseProperties: () => ({ source: "web", environment: "dev" }),
}));

vi.mock("@/services/analytics/anonymous-id", () => ({
  captureFirstLanding: vi.fn(),
  followAnalyticsConsentForIdentity: () => () => {},
  getAnonymousID: () => null,
}));

vi.mock("../posthog-consent", () => ({
  followAnalyticsConsent: () => () => {},
  forgetPostHogStorageWithoutConsent: vi.fn(),
  getConsentGatedConfig: () => ({}),
}));

describe("PostHogProvider", () => {
  beforeEach(() => {
    posthog.init.mockClear();
    posthog.register.mockClear();
    credentials.key = "phc_test";
  });

  it("registers the base properties once PostHog is initialised", async () => {
    const { PostHogProvider } = await import("../posthog-provider");

    render(<PostHogProvider>app</PostHogProvider>);

    expect(posthog.register).toHaveBeenCalledOnce();
    expect(posthog.register).toHaveBeenCalledWith({
      source: "web",
      environment: "dev",
    });
    expect(posthog.init.mock.invocationCallOrder[0]).toBeLessThan(
      posthog.register.mock.invocationCallOrder[0],
    );
  });

  it("registers nothing without a PostHog key", async () => {
    credentials.key = "";
    const { PostHogProvider } = await import("../posthog-provider");

    render(<PostHogProvider>app</PostHogProvider>);

    expect(posthog.init).not.toHaveBeenCalled();
    expect(posthog.register).not.toHaveBeenCalled();
  });
});
