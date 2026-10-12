import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useQuoteExpiry } from "../useQuoteExpiry";

afterEach(() => vi.useRealTimers());

describe("quoted price expiry", () => {
  it("stays quiet until two minutes remain and then requires a refreshed quote", () => {
    vi.useFakeTimers();
    const now = new Date("2026-10-07T12:00:00Z");
    vi.setSystemTime(now);
    const { result, unmount } = renderHook(() =>
      useQuoteExpiry(new Date(now.getTime() + 180000)),
    );
    expect(result.current).toEqual({ expired: false, expiringSoon: false });
    act(() => vi.advanceTimersByTime(61000));
    expect(result.current).toEqual({ expired: false, expiringSoon: true });
    act(() => vi.advanceTimersByTime(119000));
    expect(result.current).toEqual({ expired: true, expiringSoon: false });
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});
