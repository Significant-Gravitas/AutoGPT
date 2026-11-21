import { StrictMode } from "react";
import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { rememberConnectAttempt } from "@/lib/auth/connect-attempt";
import { useConnectFallback } from "../useConnectFallback";

let mockSearchParams = new URLSearchParams();

vi.mock("next/navigation", () => ({
  useSearchParams: () => mockSearchParams,
}));

afterEach(() => {
  mockSearchParams = new URLSearchParams();
  window.sessionStorage.clear();
});

describe("useConnectFallback", () => {
  it("keeps an unfinished sign-in's provider when its effect runs twice", async () => {
    rememberConnectAttempt("microsoft");

    // As in development, where React runs each effect twice: the second run
    // finds the note already taken.
    const { result } = renderHook(() => useConnectFallback(), {
      wrapper: StrictMode,
    });
    await act(async () => {});

    expect(result.current).toBe("microsoft");
  });

  it("names a refused sign-in's provider over an unfinished one", async () => {
    rememberConnectAttempt("microsoft");
    mockSearchParams = new URLSearchParams("connect=google");

    const { result } = renderHook(() => useConnectFallback());
    await act(async () => {});

    expect(result.current).toBe("google");
  });
});
