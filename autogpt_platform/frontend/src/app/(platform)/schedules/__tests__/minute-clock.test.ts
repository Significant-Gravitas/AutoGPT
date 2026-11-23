import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useMinuteClock } from "../useMinuteClock";

const MINUTE = 60_000;

function setVisibility(state: "visible" | "hidden") {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => state,
  });
  document.dispatchEvent(new Event("visibilitychange"));
}

describe("useMinuteClock", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-11-19T10:00:30Z"));
    setVisibility("visible");
  });
  afterEach(() => {
    vi.useRealTimers();
    setVisibility("visible");
  });

  it("moves on every minute while the page is open", () => {
    const { result } = renderHook(() => useMinuteClock());
    const first = result.current;
    expect(first).toBe(Date.parse("2026-11-19T10:00:00Z"));

    act(() => vi.advanceTimersByTime(MINUTE));
    expect(result.current).toBe(first + MINUTE);

    act(() => vi.advanceTimersByTime(2 * MINUTE));
    expect(result.current).toBe(first + 3 * MINUTE);
  });

  it("rests while the tab is hidden and catches up when it shows", () => {
    const { result } = renderHook(() => useMinuteClock());
    const first = result.current;

    act(() => setVisibility("hidden"));
    act(() => vi.advanceTimersByTime(5 * MINUTE));
    expect(vi.getTimerCount()).toBe(0);

    act(() => setVisibility("visible"));
    expect(result.current).toBe(first + 5 * MINUTE);
  });
});
