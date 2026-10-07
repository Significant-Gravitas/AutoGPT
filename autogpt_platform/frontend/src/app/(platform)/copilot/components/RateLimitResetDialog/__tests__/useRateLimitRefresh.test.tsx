import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useRateLimitRefresh } from "../useRateLimitRefresh";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("usage refresh recovery", () => {
  it("bounds an offline refresh, preserves the refusal, and recovers by retry", async () => {
    vi.useFakeTimers();
    let finishStaleRequest: (() => void) | undefined;
    const retry = vi.fn().mockReturnValueOnce(
      new Promise<void>((resolve) => {
        finishStaleRequest = resolve;
      }),
    );
    const { result, rerender } = renderHook(
      ({ message }) => useRateLimitRefresh(message, retry),
      { initialProps: { message: "Weekly limit reached" as string | null } },
    );
    expect(result.current.checking).toBe(true);
    expect(retry).toHaveBeenCalledWith(true);
    await act(() => vi.advanceTimersByTimeAsync(25_000));
    expect(result.current.checking).toBe(false);
    expect(result.current.failed).toBe(true);
    expect(result.current.guarded).toBe(true);

    rerender({ message: null });
    expect(result.current.guarded).toBe(true);
    await act(async () => finishStaleRequest?.());
    expect(result.current.failed).toBe(true);
    expect(result.current.guarded).toBe(true);
    retry.mockResolvedValueOnce(undefined);
    await act(() => result.current.refresh());
    expect(result.current.failed).toBe(false);
    expect(result.current.guarded).toBe(false);
  });

  it("releases the guard after an explicit provider switch and ignores old completion", async () => {
    vi.useFakeTimers();
    let finishRequest: (() => void) | undefined;
    const retry = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finishRequest = resolve;
        }),
    );
    const { result } = renderHook(() =>
      useRateLimitRefresh("Daily limit reached", retry),
    );
    act(() => result.current.release());
    expect(result.current.guarded).toBe(false);
    await act(async () => finishRequest?.());
    expect(result.current.guarded).toBe(false);
    expect(result.current.failed).toBe(false);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("cleans up a pending deadline on unmount", () => {
    vi.useFakeTimers();
    const { unmount } = renderHook(() =>
      useRateLimitRefresh("Daily limit reached", () => new Promise(() => {})),
    );
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});
