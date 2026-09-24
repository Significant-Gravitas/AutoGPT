import { render, renderHook } from "@testing-library/react";
import { Component, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const launchDarkly = vi.hoisted(() => ({
  flags: {} as Record<string, unknown>,
}));
const postHog = vi.hoisted(() => ({
  enabled: vi.fn(),
  payload: vi.fn(),
  capture: vi.fn(),
  // Whether /flags has answered live this page load, and how.
  loaded: true,
  errorsLoading: false,
}));
const postHogClient = vi.hoisted(() => ({
  capture: (...args: unknown[]) => postHog.capture(...args),
  featureFlags: {
    get hasLoadedFlags() {
      return postHog.loaded && !postHog.errorsLoading;
    },
  },
  onFeatureFlags: (
    callback: (
      flags: string[],
      variants: Record<string, unknown>,
      context?: { errorsLoading?: boolean },
    ) => void,
  ) => {
    if (postHog.loaded) {
      callback([], {}, { errorsLoading: postHog.errorsLoading });
    }
    return () => {};
  },
}));

vi.mock("launchdarkly-react-client-sdk", () => ({
  useFlags: () => launchDarkly.flags,
}));

vi.mock("@posthog/react", () => ({
  useFeatureFlagEnabled: (flag: string) => postHog.enabled(flag),
  useFeatureFlagPayload: (flag: string) => postHog.payload(flag),
  usePostHog: () => postHogClient,
}));

const sentry = vi.hoisted(() => ({ addFeatureFlag: vi.fn() }));

vi.mock("@sentry/nextjs", () => ({
  getClient: () => ({
    getIntegrationByName: (name: string) =>
      name === "FeatureFlags"
        ? { addFeatureFlag: sentry.addFeatureFlag }
        : undefined,
  }),
}));

vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));

const env = vi.hoisted(() => ({ launchDarklyEnabled: true }));

vi.mock("@/services/environment", () => ({
  environment: { areFeatureFlagsEnabled: () => env.launchDarklyEnabled },
}));

const HIRE_EXPERTS = "hire-experts";

describe("launchdarkly is the default backend", () => {
  it("reads the LaunchDarkly value and never touches PostHog", async () => {
    const { Flag, useGetFlag } = await loadWithBackend(undefined);
    launchDarkly.flags = { [HIRE_EXPERTS]: true };

    const { result } = renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(result.current).toBe(true);
    expect(postHog.enabled).not.toHaveBeenCalled();
    expect(postHog.payload).not.toHaveBeenCalled();
  });

  it("reports a flag LaunchDarkly has not answered for as not ready", async () => {
    const { Flag, useFlagStatus } = await loadWithBackend(undefined);
    launchDarkly.flags = {};

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));

    expect(result.current).toEqual({
      enabled: false,
      ready: false,
      answered: false,
    });
  });
});

describe("posthog backend", () => {
  it("serves an enabled flag", async () => {
    const { Flag, useFlagStatus } = await loadWithBackend("posthog");
    postHog.enabled.mockReturnValue(true);

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));

    expect(result.current).toEqual({
      enabled: true,
      ready: true,
      answered: true,
    });
  });

  it("distinguishes a conclusive off from no answer yet", async () => {
    const { Flag, useFlagStatus } = await loadWithBackend("posthog");
    postHog.enabled.mockReturnValue(false);
    const off = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));
    expect(off.result.current).toEqual({
      enabled: false,
      ready: true,
      answered: true,
    });

    postHog.loaded = false;
    const unanswered = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));
    expect(unanswered.result.current).toEqual({
      enabled: false,
      ready: false,
      answered: false,
    });
  });

  it("does not call a persisted snapshot an answer before /flags runs", async () => {
    const { Flag, useFlagStatus } = await loadWithBackend("posthog");
    postHog.loaded = false;
    postHog.enabled.mockReturnValue(true);

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));

    expect(result.current.answered).toBe(false);
  });

  it("does not call a failed /flags load an answer", async () => {
    const { Flag, useFlagStatus } = await loadWithBackend("posthog");
    postHog.errorsLoading = true;

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));

    expect(result.current.answered).toBe(false);
  });

  it("answers a flag PostHog has never heard of with its default", async () => {
    const { Flag, useFlagStatus } = await loadWithBackend("posthog");
    postHog.enabled.mockReturnValue(undefined);

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));

    expect(result.current).toEqual({
      enabled: false,
      ready: true,
      answered: true,
    });
  });

  it("serves an explicit off over a payload, as the backend does", async () => {
    const { Flag, useGetFlag } = await loadWithBackend("posthog");
    postHog.enabled.mockReturnValue(false);
    postHog.payload.mockReturnValue({ slack: true });

    const { result } = renderHook(() => useGetFlag(Flag.COPILOT_BOT_PLATFORMS));

    expect(result.current).toEqual({});
  });

  it("returns a payload for the JSON-valued flags", async () => {
    const { Flag, useGetFlag } = await loadWithBackend("posthog");
    postHog.enabled.mockReturnValue(true);
    postHog.payload.mockReturnValue({ slack: false });

    const { result } = renderHook(() => useGetFlag(Flag.COPILOT_BOT_PLATFORMS));

    expect(result.current).toEqual({ slack: false });
  });

  it("still honours a forced flag", async () => {
    const { Flag, useGetFlag } = await loadWithBackend("posthog");
    process.env.NEXT_PUBLIC_FORCE_FLAG_HIRE_EXPERTS = "true";
    postHog.enabled.mockReturnValue(false);

    const { result } = renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(result.current).toBe(true);
  });

  it("honours the force-all switch, and serves PostHog again without it", async () => {
    // The local override has to sit above the vendor choice, not inside the
    // LaunchDarkly path: force-all is how a developer opens a gate locally,
    // and picking posthog must not silently take that away.
    process.env.NEXT_PUBLIC_FORCE_ALL_FLAGS = "true";
    const forced = await loadWithBackend("posthog");
    postHog.enabled.mockReturnValue(false);
    expect(
      renderHook(() => forced.useFlagStatus(forced.Flag.HIRE_EXPERTS)).result
        .current,
    ).toEqual({ enabled: true, ready: true, answered: true });

    delete process.env.NEXT_PUBLIC_FORCE_ALL_FLAGS;
    const unforced = await loadWithBackend("posthog");
    expect(
      renderHook(() => unforced.useFlagStatus(unforced.Flag.HIRE_EXPERTS))
        .result.current,
    ).toEqual({ enabled: false, ready: true, answered: true });
  });
});

describe("dual backend", () => {
  it("serves LaunchDarkly's answer when the two disagree", async () => {
    const { Flag, useGetFlag } = await loadWithBackend("dual");
    launchDarkly.flags = { [HIRE_EXPERTS]: true };
    postHog.enabled.mockReturnValue(false);

    const { result } = renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(result.current).toBe(true);
  });

  it("reports the disagreement to PostHog and the console", async () => {
    const { Flag, useGetFlag } = await loadWithBackend("dual");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    launchDarkly.flags = { [HIRE_EXPERTS]: true };
    postHog.enabled.mockReturnValue(false);

    renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(postHog.capture).toHaveBeenCalledWith("feature_flag_mismatched", {
      flag: HIRE_EXPERTS,
      launchdarkly: { value: true, resolved: true },
      posthog: { value: false, resolved: true },
    });
    expect(warn).toHaveBeenCalled();
  });

  it("stays quiet when the two agree", async () => {
    const { Flag, useGetFlag } = await loadWithBackend("dual");
    launchDarkly.flags = { [HIRE_EXPERTS]: true };
    postHog.enabled.mockReturnValue(true);

    renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(postHog.capture).not.toHaveBeenCalled();
  });

  it("stays quiet when a JSON flag differs only in key order", async () => {
    const { Flag, useGetFlag } = await loadWithBackend("dual");
    launchDarkly.flags = {
      [Flag.COPILOT_BOT_PLATFORMS]: { slack: true, discord: false },
    };
    postHog.enabled.mockReturnValue(true);
    postHog.payload.mockReturnValue({ discord: false, slack: true });

    renderHook(() => useGetFlag(Flag.COPILOT_BOT_PLATFORMS));

    expect(postHog.capture).not.toHaveBeenCalled();
  });

  it("reports a JSON flag whose values really differ", async () => {
    const { Flag, useGetFlag } = await loadWithBackend("dual");
    vi.spyOn(console, "warn").mockImplementation(() => {});
    launchDarkly.flags = {
      [Flag.COPILOT_BOT_PLATFORMS]: { slack: true, discord: false },
    };
    postHog.enabled.mockReturnValue(true);
    postHog.payload.mockReturnValue({ discord: true, slack: true });

    renderHook(() => useGetFlag(Flag.COPILOT_BOT_PLATFORMS));

    expect(postHog.capture).toHaveBeenCalledTimes(1);
  });

  it("reports a flag LaunchDarkly has and PostHog has never heard of", async () => {
    const { Flag, useGetFlag } = await loadWithBackend("dual");
    vi.spyOn(console, "warn").mockImplementation(() => {});
    launchDarkly.flags = { [HIRE_EXPERTS]: true };
    postHog.enabled.mockReturnValue(undefined);

    renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(postHog.capture).toHaveBeenCalledWith("feature_flag_mismatched", {
      flag: HIRE_EXPERTS,
      launchdarkly: { value: true, resolved: true },
      posthog: { value: null, resolved: true },
    });
  });

  it("stays quiet while only one vendor has answered", async () => {
    // The two never resolve on the same render, so comparing before both
    // have answered reports the load order rather than a disagreement.
    const { Flag, useGetFlag } = await loadWithBackend("dual");
    vi.spyOn(console, "warn").mockImplementation(() => {});
    launchDarkly.flags = {};
    postHog.enabled.mockReturnValue(true);

    const { rerender } = renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));
    rerender();
    expect(postHog.capture).not.toHaveBeenCalled();

    launchDarkly.flags = { [HIRE_EXPERTS]: true };
    postHog.loaded = false;
    const pending = renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));
    pending.rerender();

    expect(postHog.capture).not.toHaveBeenCalled();
  });

  it("reports once LaunchDarkly catches up and still disagrees", async () => {
    const { Flag, useGetFlag } = await loadWithBackend("dual");
    vi.spyOn(console, "warn").mockImplementation(() => {});
    postHog.enabled.mockReturnValue(true);
    const { rerender } = renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));
    expect(postHog.capture).not.toHaveBeenCalled();

    launchDarkly.flags = { [HIRE_EXPERTS]: false };
    rerender();

    expect(postHog.capture).toHaveBeenCalledTimes(1);
  });

  it("serves PostHog when LaunchDarkly is not configured", async () => {
    // Dual must degrade to whichever vendor can actually answer: with no
    // LDProvider mounted, LaunchDarkly's empty flag set never resolves.
    env.launchDarklyEnabled = false;
    const { Flag, useFlagStatus } = await loadWithBackend("dual");
    launchDarkly.flags = {};
    postHog.enabled.mockReturnValue(true);

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));

    expect(result.current).toEqual({
      enabled: true,
      ready: true,
      answered: true,
    });
  });

  it("does not report a disagreement when only one vendor is configured", async () => {
    env.launchDarklyEnabled = false;
    const { Flag, useGetFlag } = await loadWithBackend("dual");
    postHog.enabled.mockReturnValue(true);

    renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(postHog.capture).not.toHaveBeenCalled();
  });
});

describe("Sentry's flag context", () => {
  it.each([
    ["launchdarkly", undefined],
    ["posthog", "posthog"],
    ["dual", "dual"],
  ])("records the served boolean in %s mode", async (_, backend) => {
    const { Flag, useFlagStatus } = await loadWithBackend(backend);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    launchDarkly.flags = { [HIRE_EXPERTS]: true };
    postHog.enabled.mockReturnValue(backend === "posthog");

    renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));

    expect(sentry.addFeatureFlag).toHaveBeenCalledExactlyOnceWith(
      HIRE_EXPERTS,
      true,
    );
  });

  it("records a flag read by a component that throws in the same render", async () => {
    const { Flag, useGetFlag } = await loadWithBackend(undefined);
    vi.spyOn(console, "error").mockImplementation(() => {});
    launchDarkly.flags = { [HIRE_EXPERTS]: true };
    let recordedAtCatch: unknown[][] = [];
    function Reader(): ReactNode {
      useGetFlag(Flag.HIRE_EXPERTS);
      throw new Error("render failed");
    }
    class Boundary extends Component<{ children: ReactNode }> {
      state = { failed: false };
      static getDerivedStateFromError() {
        return { failed: true };
      }
      componentDidCatch() {
        recordedAtCatch = [...sentry.addFeatureFlag.mock.calls];
      }
      render() {
        return this.state.failed ? null : this.props.children;
      }
    }

    render(
      <Boundary>
        <Reader />
      </Boundary>,
    );

    expect(recordedAtCatch).toContainEqual([HIRE_EXPERTS, true]);
  });

  it("records the default it serves while flags are disabled", async () => {
    env.launchDarklyEnabled = false;
    const { Flag, useGetFlag } = await loadWithBackend(undefined);
    launchDarkly.flags = { [HIRE_EXPERTS]: true };

    const { result } = renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(result.current).toBe(false);
    expect(sentry.addFeatureFlag).toHaveBeenCalledExactlyOnceWith(
      HIRE_EXPERTS,
      false,
    );
  });

  it("records nothing for an env-forced flag", async () => {
    const { Flag, useFlagStatus, useGetFlag } =
      await loadWithBackend(undefined);
    process.env.NEXT_PUBLIC_FORCE_FLAG_HIRE_EXPERTS = "true";

    renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));
    renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(sentry.addFeatureFlag).not.toHaveBeenCalled();
  });

  it("records nothing for a JSON-valued flag", async () => {
    const { Flag, useGetFlag } = await loadWithBackend(undefined);
    launchDarkly.flags = { "copilot-bot-platforms": { slack: false } };

    renderHook(() => useGetFlag(Flag.COPILOT_BOT_PLATFORMS));

    expect(sentry.addFeatureFlag).not.toHaveBeenCalled();
  });

  it("still serves the flag when recording throws", async () => {
    const { Flag, useGetFlag } = await loadWithBackend(undefined);
    sentry.addFeatureFlag.mockImplementation(() => {
      throw new Error("sentry down");
    });
    launchDarkly.flags = { [HIRE_EXPERTS]: true };

    const debug = vi.spyOn(console, "debug").mockImplementation(() => {});

    const { result } = renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(result.current).toBe(true);
    expect(debug).toHaveBeenCalledOnce();
  });
});

describe("posthog flags follow the provider's gate", () => {
  it("falls back to defaults outside cloud, where no PostHogProvider mounts", async () => {
    process.env.NEXT_PUBLIC_BEHAVE_AS = "LOCAL";
    const { Flag, useFlagStatus } = await loadWithBackend("posthog");
    postHog.enabled.mockReturnValue(true);

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));

    expect(result.current).toEqual({
      enabled: false,
      ready: true,
      answered: true,
    });
  });
});

beforeEach(() => {
  launchDarkly.flags = {};
  postHog.enabled.mockReturnValue(undefined);
  postHog.payload.mockReturnValue(undefined);
  postHog.capture.mockClear();
  sentry.addFeatureFlag.mockReset();
  postHog.loaded = true;
  postHog.errorsLoading = false;
  Object.keys(process.env)
    .filter((key) => key.startsWith("NEXT_PUBLIC_FORCE_FLAG_"))
    .forEach((key) => delete process.env[key]);
  delete process.env.NEXT_PUBLIC_FORCE_ALL_FLAGS;
  env.launchDarklyEnabled = true;
  process.env.NEXT_PUBLIC_BEHAVE_AS = "CLOUD";
  process.env.NEXT_PUBLIC_POSTHOG_KEY = "phc_test";
  process.env.NEXT_PUBLIC_POSTHOG_HOST = "https://eu.i.posthog.com";
});

afterEach(() => {
  vi.restoreAllMocks();
});

// The backend is read once at module load, so each case needs a fresh import.
async function loadWithBackend(backend: string | undefined) {
  if (backend === undefined) {
    delete process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND;
  } else {
    process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND = backend;
  }
  vi.resetModules();
  return import("../use-get-flag");
}
