import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

interface InitConfig {
  loaded: (client: { setPersonPropertiesForFlags: unknown }) => void;
  before_send: (event: unknown) => unknown;
  [option: string]: unknown;
}

const posthog = vi.hoisted(() => ({
  init: vi.fn((_token: string, _config: InitConfig, _name?: string) => ({})),
  register: vi.fn(),
}));
const credentials = vi.hoisted(() => ({
  key: "phc_test" as string,
  host: "https://eu.i.posthog.com",
  flagsKey: undefined as string | undefined,
}));
const flagBackend = vi.hoisted(() => ({ usesPostHog: true }));

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
      host: credentials.host,
      flagsKey: credentials.flagsKey,
    }),
  },
}));

vi.mock("@/services/feature-flags/flag-backend", () => ({
  usesPostHog: () => flagBackend.usesPostHog,
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
    // The flags-only instance is module state; each case starts without one.
    vi.resetModules();
    posthog.init.mockClear();
    posthog.register.mockClear();
    credentials.key = "phc_test";
    credentials.host = "https://eu.i.posthog.com";
    credentials.flagsKey = undefined;
    flagBackend.usesPostHog = true;
  });

  it("sends PostHog Cloud traffic through our own domain", async () => {
    const { PostHogProvider } = await import("../posthog-provider");

    render(<PostHogProvider>app</PostHogProvider>);

    expect(posthog.init).toHaveBeenCalledWith(
      "phc_test",
      expect.objectContaining({
        api_host: "/relay/pnosthog",
        ui_host: "https://eu.posthog.com",
      }),
    );
  });

  it("talks to a self-hosted PostHog directly", async () => {
    credentials.host = "https://posthog.example.com";
    const { PostHogProvider } = await import("../posthog-provider");

    render(<PostHogProvider>app</PostHogProvider>);

    const config = posthog.init.mock.calls[0][1];
    expect(config.api_host).toBe("https://posthog.example.com");
    expect(config).not.toHaveProperty("ui_host");
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

  it("gives flag evaluation the environment before the first /flags call", async () => {
    const { PostHogProvider } = await import("../posthog-provider");

    render(<PostHogProvider>app</PostHogProvider>);

    const client = { setPersonPropertiesForFlags: vi.fn() };
    posthog.init.mock.calls[0][1].loaded(client);
    expect(client.setPersonPropertiesForFlags).toHaveBeenCalledWith(
      { environment: "dev" },
      false,
    );
  });

  describe("the PostHog project flags are read from", () => {
    it.each([
      ["no flags key is set", undefined],
      ["the flags key is the analytics key", "phc_test"],
    ])("is the analytics project when %s", async (_, flagsKey) => {
      credentials.flagsKey = flagsKey;
      const { PostHogProvider } = await import("../posthog-provider");

      render(<PostHogProvider>app</PostHogProvider>);

      expect(posthog.init).toHaveBeenCalledOnce();
    });

    it("is read through a flags-only instance when the flags key names another project", async () => {
      credentials.key = "phc_dev";
      credentials.flagsKey = "phc_prod";
      const { PostHogProvider } = await import("../posthog-provider");

      render(<PostHogProvider>app</PostHogProvider>);

      expect(posthog.init).toHaveBeenCalledTimes(2);
      const [token, config, name] = posthog.init.mock.calls[1];
      expect(token).toBe("phc_prod");
      expect(name).toBe("flags");
      expect(config).toMatchObject({
        api_host: "/relay/pnosthog",
        persistence: "memory",
        opt_out_capturing_by_default: true,
        autocapture: false,
        capture_pageview: false,
        disable_session_recording: true,
      });
      expect(config.before_send({ event: "$feature_flag_called" })).toBeNull();
    });

    it("needs no flags-only instance while LaunchDarkly answers flags", async () => {
      flagBackend.usesPostHog = false;
      credentials.key = "phc_dev";
      credentials.flagsKey = "phc_prod";
      const { PostHogProvider } = await import("../posthog-provider");

      render(<PostHogProvider>app</PostHogProvider>);

      expect(posthog.init).toHaveBeenCalledOnce();
    });
  });
});
