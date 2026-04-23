import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useElapsedTimer } from "../useElapsedTimer";

describe("useElapsedTimer", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-23T10:00:00.000Z"));
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts at zero and ticks once per second while running", () => {
    const { result } = renderHook(() => useElapsedTimer(true));
    expect(result.current.elapsedSeconds).toBe(0);
    act(() => vi.advanceTimersByTime(3000));
    expect(result.current.elapsedSeconds).toBe(3);
  });

  it("stops ticking and resets when isRunning flips to false", () => {
    const { result, rerender } = renderHook(
      ({ running }) => useElapsedTimer(running),
      { initialProps: { running: true } },
    );
    act(() => vi.advanceTimersByTime(2000));
    expect(result.current.elapsedSeconds).toBe(2);
    rerender({ running: false });
    act(() => vi.advanceTimersByTime(5000));
    // No ticks after stop — elapsed stays at last reading until the next
    // `running:true` transition, which re-anchors to the current time.
    expect(result.current.elapsedSeconds).toBe(2);
  });

  it("anchors to an ISO timestamp so a fresh mount reflects real elapsed time", () => {
    // Anchor 15s in the past relative to the mocked system time.
    const anchor = new Date("2026-04-23T09:59:45.000Z").toISOString();
    const { result } = renderHook(() => useElapsedTimer(true, anchor));
    expect(result.current.elapsedSeconds).toBe(15);
    act(() => vi.advanceTimersByTime(5000));
    expect(result.current.elapsedSeconds).toBe(20);
  });

  it("clamps a future-dated anchor to zero rather than a negative seconds count", () => {
    const anchor = new Date("2026-04-23T10:00:10.000Z").toISOString();
    const { result } = renderHook(() => useElapsedTimer(true, anchor));
    expect(result.current.elapsedSeconds).toBe(0);
  });

  it("falls back to mount-time counting when anchor is invalid", () => {
    const { result } = renderHook(() => useElapsedTimer(true, "not-a-date"));
    expect(result.current.elapsedSeconds).toBe(0);
    act(() => vi.advanceTimersByTime(4000));
    expect(result.current.elapsedSeconds).toBe(4);
  });

  it("re-syncs when a late-arriving anchor replaces the previous one mid-run", () => {
    // Simulate the real case: timer mounts with no anchor (session data
    // hasn't loaded yet), starts counting from mount.  Then the session
    // query resolves and surfaces the actual server timestamp, which should
    // correct the elapsed reading rather than being ignored.
    const { result, rerender } = renderHook(
      ({ anchor }: { anchor: string | null }) => useElapsedTimer(true, anchor),
      { initialProps: { anchor: null as string | null } },
    );
    act(() => vi.advanceTimersByTime(2000));
    expect(result.current.elapsedSeconds).toBe(2);

    rerender({
      anchor: new Date("2026-04-23T09:59:00.000Z").toISOString(),
    });
    // Clock is at 10:00:02 (after the 2s advance), anchor is 60s earlier,
    // so elapsed jumps to 62.
    expect(result.current.elapsedSeconds).toBe(62);
  });
});
