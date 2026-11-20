import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
const mocks = vi.hoisted(() => ({ usage: vi.fn() }));
vi.mock("@/services/usageExperience/useUsageExperience", () => ({
  useUsageExperience: mocks.usage,
}));
import { useIsUsageLimitReached } from "../useIsUsageLimitReached";
beforeEach(() => {
  mocks.usage.mockReturnValue({
    experience: { blocked: true },
    isError: false,
  });
});
describe("chat usage gating", () => {
  it("keeps platform-funded messages blocked after the allowance is exhausted", () => {
    expect(
      renderHook(() => useIsUsageLimitReached("platform")).result.current,
    ).toBe(true);
  });
  it("unblocks the same conversation only after a confirmed provider switch", () => {
    const { result, rerender } = renderHook(
      ({ provider }) => useIsUsageLimitReached(provider),
      { initialProps: { provider: "platform" } },
    );
    expect(result.current).toBe(true);
    rerender({ provider: "codex" });
    expect(result.current).toBe(false);
  });
  it("lets a Microsoft 365 Copilot chat carry on, which its own plan pays for", () => {
    expect(
      renderHook(() => useIsUsageLimitReached("microsoft_365_copilot")).result
        .current,
    ).toBe(false);
  });
  it.each([null, undefined, "some_new_provider"])(
    "does not infer a provider exemption from %s",
    (provider) => {
      expect(
        renderHook(() => useIsUsageLimitReached(provider)).result.current,
      ).toBe(true);
    },
  );
  it("pauses platform sending while usage is unavailable", () => {
    mocks.usage.mockReturnValue({
      experience: { blocked: false },
      isError: true,
    });
    expect(
      renderHook(() => useIsUsageLimitReached("platform")).result.current,
    ).toBe(true);
  });
});
