import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const launchDarkly = vi.hoisted(() => ({
  flags: {} as Record<string, unknown>,
}));
const postHog = vi.hoisted(() => ({
  enabled: vi.fn(),
  payload: vi.fn(),
  capture: vi.fn(),
}));

vi.mock("launchdarkly-react-client-sdk", () => ({
  useFlags: () => launchDarkly.flags,
}));

vi.mock("@posthog/react", () => ({
  useFeatureFlagEnabled: (flag: string) => postHog.enabled(flag),
  useFeatureFlagPayload: (flag: string) => postHog.payload(flag),
  usePostHog: () => ({ capture: postHog.capture }),
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

    expect(result.current).toEqual({ enabled: false, ready: false });
  });
});

describe("posthog backend", () => {
  it("serves an enabled flag", async () => {
    const { Flag, useFlagStatus } = await loadWithBackend("posthog");
    postHog.enabled.mockReturnValue(true);

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));

    expect(result.current).toEqual({ enabled: true, ready: true });
  });

  it("distinguishes a conclusive off from no answer yet", async () => {
    const { Flag, useFlagStatus } = await loadWithBackend("posthog");
    postHog.enabled.mockReturnValue(false);
    const off = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));
    expect(off.result.current).toEqual({ enabled: false, ready: true });

    postHog.enabled.mockReturnValue(undefined);
    const unanswered = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));
    expect(unanswered.result.current).toEqual({ enabled: false, ready: false });
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

    expect(postHog.capture).toHaveBeenCalledWith("feature_flag_mismatch", {
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

    expect(result.current).toEqual({ enabled: true, ready: true });
  });

  it("does not report a disagreement when only one vendor is configured", async () => {
    env.launchDarklyEnabled = false;
    const { Flag, useGetFlag } = await loadWithBackend("dual");
    postHog.enabled.mockReturnValue(true);

    renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(postHog.capture).not.toHaveBeenCalled();
  });
});

describe("posthog flags follow the provider's gate", () => {
  it("falls back to defaults outside cloud, where no PostHogProvider mounts", async () => {
    process.env.NEXT_PUBLIC_BEHAVE_AS = "LOCAL";
    const { Flag, useFlagStatus } = await loadWithBackend("posthog");
    postHog.enabled.mockReturnValue(true);

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));

    expect(result.current).toEqual({ enabled: false, ready: true });
  });
});

beforeEach(() => {
  launchDarkly.flags = {};
  postHog.enabled.mockReturnValue(undefined);
  postHog.payload.mockReturnValue(undefined);
  postHog.capture.mockClear();
  Object.keys(process.env)
    .filter((key) => key.startsWith("NEXT_PUBLIC_FORCE_FLAG_"))
    .forEach((key) => delete process.env[key]);
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
