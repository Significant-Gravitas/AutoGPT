import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type FlagsCallback = (
  flags: string[],
  variants: Record<string, unknown>,
  context?: { errorsLoading?: boolean },
) => void;

// Dev's analytics project has never heard of the flag and has already loaded
// its flags (posthog-js keeps them from the last visit), while prod's flags
// project has it on and is still loading.
const analytics = vi.hoisted(() =>
  fakeClient({ loaded: true, flag: undefined }),
);
const flagsOnly = vi.hoisted(() => fakeClient({ loaded: false, flag: true }));
const posthog = vi.hoisted(() => ({
  ...analytics,
  init: vi.fn((_token: string, _config: unknown, name?: string) =>
    name === "flags" ? flagsOnly : undefined,
  ),
  register: vi.fn(),
}));

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
      key: "phc_dev",
      host: "https://eu.i.posthog.com",
    }),
  },
}));

vi.mock("@/services/feature-flags/flag-backend", () => ({
  usesPostHog: () => true,
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

describe("a flag read in the first render, with flags from another project", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", "phc_dev");
    vi.stubEnv("NEXT_PUBLIC_POSTHOG_HOST", "https://eu.i.posthog.com");
    vi.stubEnv("NEXT_PUBLIC_POSTHOG_FLAGS_KEY", "phc_prod");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("never answers from the analytics project, and resolves with the flags project's answer", async () => {
    const { PostHogProvider } = await import("../posthog-provider");
    const { usePostHogFlag } = await import(
      "@/services/feature-flags/backends/posthog"
    );
    const answers: { value: unknown; resolved: boolean }[] = [];
    function FlagReader() {
      const answer = usePostHogFlag("graphiti-memory");
      answers.push(answer);
      return <p>{answer.resolved ? String(answer.value) : "loading"}</p>;
    }

    render(
      <PostHogProvider>
        <FlagReader />
      </PostHogProvider>,
    );
    act(() => flagsOnly.finishLoading());

    expect(screen.getByText("true")).toBeDefined();
    expect(answers.filter((answer) => answer.resolved)).not.toContainEqual(
      expect.objectContaining({ value: undefined }),
    );
    expect(answers[0]).toEqual({ value: undefined, resolved: false });
  });
});

function fakeClient({ loaded, flag }: { loaded: boolean; flag?: boolean }) {
  const callbacks = new Set<FlagsCallback>();
  const state = { loaded };
  return {
    featureFlags: {
      get hasLoadedFlags() {
        return state.loaded;
      },
    },
    isFeatureEnabled: () => (state.loaded ? flag : undefined),
    getFeatureFlagPayload: () => undefined,
    // posthog-js answers a late subscriber at once when flags are loaded.
    onFeatureFlags(callback: FlagsCallback) {
      callbacks.add(callback);
      if (state.loaded) callback([], {});
      return () => callbacks.delete(callback);
    },
    finishLoading() {
      state.loaded = true;
      callbacks.forEach((callback) => callback([], {}));
    },
  };
}
