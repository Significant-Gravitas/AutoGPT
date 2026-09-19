import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Flag, useFlagStatus, useGetFlag } from "../use-get-flag";

// Phase 2 replaces 75 `vi.mock("launchdarkly-react-client-sdk")` blocks with
// this one, vendor-independent, seam. This test is what keeps it a seam.
const source = vi.hoisted(() => ({
  results: {} as Record<string, { value: unknown; resolved: boolean }>,
}));

vi.mock("../flag-source", () => ({
  useFlagSource: (key: string) =>
    source.results[key] ?? { value: undefined, resolved: false },
}));

vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));

vi.mock("@/services/environment", () => ({
  environment: { areFeatureFlagsEnabled: () => true },
}));

describe("mocking the flag source controls every flag hook", () => {
  it("drives useGetFlag without mocking any vendor SDK", () => {
    source.results = { "hire-experts": { value: true, resolved: true } };

    const { result } = renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(result.current).toBe(true);
  });

  it("drives useFlagStatus, including the not-answered-yet case", () => {
    source.results = {};

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));

    expect(result.current).toEqual({ enabled: false, ready: false });
  });
});
