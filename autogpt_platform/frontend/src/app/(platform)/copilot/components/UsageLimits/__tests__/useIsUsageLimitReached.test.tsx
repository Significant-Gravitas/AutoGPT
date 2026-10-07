import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
const mocks = vi.hoisted(() => ({ usage: vi.fn(), session: vi.fn() }));
vi.mock("@/services/usageExperience/useUsageExperience", () => ({
  useUsageExperience: mocks.usage,
}));
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useGetV2GetSession: mocks.session,
}));
import { useIsUsageLimitReached } from "../useIsUsageLimitReached";
beforeEach(() => {
  mocks.usage.mockReturnValue({
    experience: { blocked: true },
    isError: false,
  });
  mocks.session.mockReturnValue({
    data: {
      status: 200,
      data: { metadata: { llm_auth_provider: "platform" } },
    },
  });
});
describe("chat usage gating", () => {
  it("keeps platform-funded messages blocked after the allowance is exhausted", () => {
    expect(
      renderHook(() => useIsUsageLimitReached("chat")).result.current,
    ).toBe(true);
  });
  it("unblocks the same conversation only after a confirmed provider switch", () => {
    const { result, rerender } = renderHook(() =>
      useIsUsageLimitReached("chat"),
    );
    expect(result.current).toBe(true);
    mocks.session.mockReturnValue({
      data: { status: 200, data: { metadata: { llm_auth_provider: "codex" } } },
    });
    rerender();
    expect(result.current).toBe(false);
  });
  it("does not infer a provider exemption from missing session data", () => {
    mocks.session.mockReturnValue({ data: undefined });
    expect(
      renderHook(() => useIsUsageLimitReached("chat")).result.current,
    ).toBe(true);
  });
  it("pauses platform sending while usage is unavailable", () => {
    mocks.usage.mockReturnValue({
      experience: { blocked: false },
      isError: true,
    });
    expect(
      renderHook(() => useIsUsageLimitReached("chat")).result.current,
    ).toBe(true);
  });
});
