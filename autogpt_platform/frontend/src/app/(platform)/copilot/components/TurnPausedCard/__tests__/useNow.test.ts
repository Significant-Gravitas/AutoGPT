import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useNow } from "../components/LivePausedCard/useNow";

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("useNow", () => {
  it("keeps time until the moment it counts to, then stops", () => {
    const start = Date.now();
    const { result } = renderHook(() => useNow(start + 2_500));

    act(() => vi.advanceTimersByTime(1_000));
    expect(result.current.getTime()).toBe(start + 1_000);

    act(() => vi.advanceTimersByTime(2_000));
    const stopped = result.current.getTime();
    expect(stopped).toBeGreaterThanOrEqual(start + 2_500);

    act(() => vi.advanceTimersByTime(5_000));
    expect(result.current.getTime()).toBe(stopped);
  });

  it("doesn't tick with nothing to count to", () => {
    const { result } = renderHook(() => useNow(null));
    const first = result.current.getTime();

    act(() => vi.advanceTimersByTime(5_000));

    expect(result.current.getTime()).toBe(first);
  });
});
