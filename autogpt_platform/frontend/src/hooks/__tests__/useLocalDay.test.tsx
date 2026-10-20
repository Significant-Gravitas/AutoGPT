import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useLocalDay } from "../useLocalDay";

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date(2026, 9, 19, 23, 58, 30));
});

afterEach(() => {
  vi.useRealTimers();
});

describe("useLocalDay", () => {
  it("re-renders once when the local day changes, and not before", () => {
    let renders = 0;
    const { result } = renderHook(() => {
      renders += 1;
      return useLocalDay();
    });
    const today = result.current;
    const rendersBefore = renders;

    act(() => {
      vi.advanceTimersByTime(60_000);
    });
    expect(result.current).toBe(today);
    expect(renders).toBe(rendersBefore);

    act(() => {
      vi.advanceTimersByTime(60_000);
    });
    expect(result.current).not.toBe(today);
    expect(result.current).toBe(new Date(2026, 9, 20).toDateString());
    expect(renders).toBe(rendersBefore + 1);
  });
});
