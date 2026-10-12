import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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
      JSON.stringify({ "hire-experts": true }),
    );
    const { Flag, useGetFlag, useFlagStatus } = await loadFlags();

    expect(renderHook(() => useGetFlag(Flag.HIRE_EXPERTS)).result.current).toBe(
      true,
    );
    expect(
      renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS)).result.current,
    ).toEqual({ enabled: true, ready: true, answered: true });
    expect(renderHook(() => useGetFlag(Flag.SKILLS_HUB)).result.current).toBe(
      false,
    );
  });

  it("keeps route gates waiting until hydration can read the overrides", async () => {
    vi.stubEnv("NEXT_PUBLIC_PW_TEST", "true");
    const { Flag, useFlagStatus } = await loadFlags();
    const { renderToString } = await import("react-dom/server");

    let status: ReturnType<typeof useFlagStatus> | undefined;
    function Probe() {
      status = useFlagStatus(Flag.HIRE_EXPERTS);
      return null;
    }

    renderToString(<Probe />);
    expect(status).toMatchObject({ ready: false, answered: false });
  });

  it("is ignored outside a Playwright build", async () => {
    vi.stubEnv("NEXT_PUBLIC_PW_TEST", "false");
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ "hire-experts": true }),
    );
    const { Flag, useGetFlag } = await loadFlags();

    expect(renderHook(() => useGetFlag(Flag.HIRE_EXPERTS)).result.current).toBe(
      false,
    );
  });

  it("falls back to the default for a value of the wrong type or bad JSON", async () => {
    vi.stubEnv("NEXT_PUBLIC_PW_TEST", "true");
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ "hire-experts": "yes" }),
    );
    const { Flag, useGetFlag } = await loadFlags();

    expect(renderHook(() => useGetFlag(Flag.HIRE_EXPERTS)).result.current).toBe(
      false,
    );

    window.localStorage.setItem(STORAGE_KEY, "{not json");
    expect(renderHook(() => useGetFlag(Flag.HIRE_EXPERTS)).result.current).toBe(
      false,
    );
  });

  it("wins over a build-time env override", async () => {
    vi.stubEnv("NEXT_PUBLIC_PW_TEST", "true");
    vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_HIRE_EXPERTS", "true");
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ "hire-experts": false }),
    );
    const { Flag, useGetFlag } = await loadFlags();

    expect(renderHook(() => useGetFlag(Flag.HIRE_EXPERTS)).result.current).toBe(
      false,
    );
  });
});
