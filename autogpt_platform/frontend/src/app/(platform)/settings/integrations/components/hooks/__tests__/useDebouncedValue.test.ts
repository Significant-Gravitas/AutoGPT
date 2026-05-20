import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { useDebouncedValue } from "../useDebouncedValue";

describe("useDebouncedValue", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  test("returns the initial value synchronously", () => {
    const { result } = renderHook(() => useDebouncedValue("seed", 200));
    expect(result.current).toBe("seed");
  });

  test("does not change before the delay elapses", () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebouncedValue(value, 200),
      { initialProps: { value: "a" } },
    );
    rerender({ value: "b" });
    act(() => {
      vi.advanceTimersByTime(199);
    });
    expect(result.current).toBe("a");
  });

  test("updates to the latest value once the delay elapses", () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebouncedValue(value, 200),
      { initialProps: { value: "a" } },
    );
    rerender({ value: "b" });
    act(() => {
      vi.advanceTimersByTime(200);
    });
    expect(result.current).toBe("b");
  });

  test("only the last rapid change wins (timer is reset on each rerender)", () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebouncedValue(value, 200),
      { initialProps: { value: "a" } },
    );
    rerender({ value: "b" });
    act(() => {
      vi.advanceTimersByTime(150);
    });
    rerender({ value: "c" });
    act(() => {
      vi.advanceTimersByTime(150);
    });
    // 300ms total but only 150ms since the last change.
    expect(result.current).toBe("a");
    act(() => {
      vi.advanceTimersByTime(50);
    });
    expect(result.current).toBe("c");
  });
});
