import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AvatarStatus } from "../helpers";
import { useExpression } from "../useExpression";

describe("useExpression", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0);
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it.each([
    { status: "idle", isLive: false },
    { status: "sleeping", isLive: true },
    { status: "failed", isLive: true },
    { status: "working", isLive: true },
  ] as { status: AvatarStatus; isLive: boolean }[])(
    "opens the eyes when interrupted by $status / live=$isLive",
    (next) => {
      const { result, rerender, unmount } = renderHook(useExpression, {
        initialProps: { status: "idle" as AvatarStatus, isLive: true },
      });
      act(() => vi.advanceTimersByTime(5000));
      expect(result.current.isBlinking).toBe(true);
      rerender(next);
      expect(result.current.isBlinking).toBe(false);
      unmount();
      expect(vi.getTimerCount()).toBe(0);
    },
  );

  it("finishes and reschedules a live blink", () => {
    const { result } = renderHook(() =>
      useExpression({ status: "idle", isLive: true }),
    );
    act(() => vi.advanceTimersByTime(5000));
    expect(result.current.isBlinking).toBe(true);
    act(() => vi.advanceTimersByTime(130));
    expect(result.current.isBlinking).toBe(false);
    act(() => vi.advanceTimersByTime(5000));
    expect(result.current.isBlinking).toBe(true);
  });
});
