import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// These read LaunchDarkly; `flag-backend` picks the vendor once, at load.
vi.hoisted(() => {
  process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND = "launchdarkly";
});

vi.mock("launchdarkly-react-client-sdk", () => ({
  useFlags: () => ({}),
}));

vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));

vi.mock("@/services/environment", () => ({
  environment: { areFeatureFlagsEnabled: () => false },
}));

const STORAGE_KEY = "autogpt-pw-flag-overrides";

async function loadFlags() {
  vi.resetModules();
  return import("../use-get-flag");
}

describe("Playwright runtime flag overrides", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    window.localStorage.clear();
  });

  it("serves the stored value in a Playwright build", async () => {
    vi.stubEnv("NEXT_PUBLIC_PW_TEST", "true");
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ "generic-trigger-agents": true }),
    );
    const { Flag, useGetFlag, useFlagStatus } = await loadFlags();

    expect(
      renderHook(() => useGetFlag(Flag.GENERIC_TRIGGER_AGENTS)).result.current,
    ).toBe(true);
    expect(
      renderHook(() => useFlagStatus(Flag.GENERIC_TRIGGER_AGENTS)).result
        .current,
    ).toEqual({ enabled: true, ready: true, answered: true });
    expect(
      renderHook(() => useGetFlag(Flag.CHAT_WORKSPACE_FILES)).result.current,
    ).toBe(false);
  });

  it("keeps route gates waiting until hydration can read the overrides", async () => {
    vi.stubEnv("NEXT_PUBLIC_PW_TEST", "true");
    const { Flag, useFlagStatus } = await loadFlags();
    const { renderToString } = await import("react-dom/server");

    let status: ReturnType<typeof useFlagStatus> | undefined;
    function Probe() {
      status = useFlagStatus(Flag.GENERIC_TRIGGER_AGENTS);
      return null;
    }

    renderToString(<Probe />);
    expect(status).toMatchObject({ ready: false, answered: false });
  });

  it("is ignored outside a Playwright build", async () => {
    vi.stubEnv("NEXT_PUBLIC_PW_TEST", "false");
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ "generic-trigger-agents": true }),
    );
    const { Flag, useGetFlag } = await loadFlags();

    expect(
      renderHook(() => useGetFlag(Flag.GENERIC_TRIGGER_AGENTS)).result.current,
    ).toBe(false);
  });

  it("falls back to the default for a value of the wrong type or bad JSON", async () => {
    vi.stubEnv("NEXT_PUBLIC_PW_TEST", "true");
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ "generic-trigger-agents": "yes" }),
    );
    const { Flag, useGetFlag } = await loadFlags();

    expect(
      renderHook(() => useGetFlag(Flag.GENERIC_TRIGGER_AGENTS)).result.current,
    ).toBe(false);

    window.localStorage.setItem(STORAGE_KEY, "{not json");
    expect(
      renderHook(() => useGetFlag(Flag.GENERIC_TRIGGER_AGENTS)).result.current,
    ).toBe(false);
  });

  it("wins over a build-time env override", async () => {
    vi.stubEnv("NEXT_PUBLIC_PW_TEST", "true");
    vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_GENERIC_TRIGGER_AGENTS", "true");
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ "generic-trigger-agents": false }),
    );
    const { Flag, useGetFlag } = await loadFlags();

    expect(
      renderHook(() => useGetFlag(Flag.GENERIC_TRIGGER_AGENTS)).result.current,
    ).toBe(false);
  });
});
