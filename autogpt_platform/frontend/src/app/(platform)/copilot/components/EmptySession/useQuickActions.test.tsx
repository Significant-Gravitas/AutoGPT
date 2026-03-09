import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { User } from "@supabase/supabase-js";
import { DEFAULT_QUICK_ACTIONS } from "./helpers";
import { useQuickActions } from "./useQuickActions";

const { mockUseGetV1GetBusinessUnderstandingPrompts } = vi.hoisted(() => ({
  mockUseGetV1GetBusinessUnderstandingPrompts: vi.fn(),
}));

vi.mock("@/app/api/__generated__/endpoints/auth/auth", () => ({
  useGetV1GetBusinessUnderstandingPrompts:
    mockUseGetV1GetBusinessUnderstandingPrompts,
}));

function makeUser() {
  return { id: "user-1" } as User;
}

describe("useQuickActions", () => {
  it("uses server prompts when available", () => {
    mockUseGetV1GetBusinessUnderstandingPrompts.mockReturnValue({
      data: ["Help me automate onboarding", "Find my biggest bottleneck"],
    });

    const { result } = renderHook(() => useQuickActions(makeUser()));

    expect(result.current).toEqual([
      "Help me automate onboarding",
      "Find my biggest bottleneck",
    ]);
    expect(mockUseGetV1GetBusinessUnderstandingPrompts).toHaveBeenCalledWith({
      query: expect.objectContaining({ enabled: true }),
    });
  });

  it("falls back to defaults when the user is not authenticated", () => {
    mockUseGetV1GetBusinessUnderstandingPrompts.mockReturnValue({
      data: undefined,
    });

    const { result } = renderHook(() => useQuickActions(null));

    expect(result.current).toEqual(DEFAULT_QUICK_ACTIONS);
    expect(mockUseGetV1GetBusinessUnderstandingPrompts).toHaveBeenCalledWith({
      query: expect.objectContaining({ enabled: false }),
    });
  });

  it("falls back to defaults when the API returns no prompts", () => {
    mockUseGetV1GetBusinessUnderstandingPrompts.mockReturnValue({
      data: [],
      error: new Error("no prompts"),
      isError: true,
    });

    const { result } = renderHook(() => useQuickActions(makeUser()));

    expect(result.current).toEqual(DEFAULT_QUICK_ACTIONS);
  });
});
