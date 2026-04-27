import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useStreamActivityWatchdog } from "../useStreamActivityWatchdog";

const STREAM_STALL_TIMEOUT_MS = 60_000;

describe("useStreamActivityWatchdog", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  function setup(
    overrides: Partial<{
      status: "submitted" | "streaming" | "ready" | "error";
      sessionId: string | null;
      isReconnectScheduled: boolean;
      isUserStopping: boolean;
      activityToken: string | number;
    }> = {},
  ) {
    const handleReconnect = vi.fn();
    const isUserStoppingRef = { current: overrides.isUserStopping ?? false };
    const handleReconnectRef = { current: handleReconnect };
    const args = {
      sessionId: "sessionId" in overrides ? overrides.sessionId! : "sid",
      status: overrides.status ?? "streaming",
      activityToken: overrides.activityToken ?? "1:0",
      isReconnectScheduled: overrides.isReconnectScheduled ?? false,
      isUserStoppingRef,
      handleReconnectRef,
    };
    return { handleReconnect, args };
  }

  it("fires reconnect after stall timeout while streaming", () => {
    const { handleReconnect, args } = setup({ status: "streaming" });
    renderHook(() => useStreamActivityWatchdog(args));
    vi.advanceTimersByTime(STREAM_STALL_TIMEOUT_MS - 1);
    expect(handleReconnect).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(handleReconnect).toHaveBeenCalledWith("sid");
  });

  it("does not fire when status is ready", () => {
    const { handleReconnect, args } = setup({ status: "ready" });
    renderHook(() => useStreamActivityWatchdog(args));
    vi.advanceTimersByTime(STREAM_STALL_TIMEOUT_MS * 2);
    expect(handleReconnect).not.toHaveBeenCalled();
  });

  it("does not fire when sessionId is null", () => {
    const { handleReconnect, args } = setup({ sessionId: null });
    renderHook(() => useStreamActivityWatchdog(args));
    vi.advanceTimersByTime(STREAM_STALL_TIMEOUT_MS * 2);
    expect(handleReconnect).not.toHaveBeenCalled();
  });

  it("does not fire when reconnect is already scheduled", () => {
    const { handleReconnect, args } = setup({ isReconnectScheduled: true });
    renderHook(() => useStreamActivityWatchdog(args));
    vi.advanceTimersByTime(STREAM_STALL_TIMEOUT_MS * 2);
    expect(handleReconnect).not.toHaveBeenCalled();
  });

  it("does not fire when user is stopping", () => {
    const { handleReconnect, args } = setup({ isUserStopping: true });
    renderHook(() => useStreamActivityWatchdog(args));
    vi.advanceTimersByTime(STREAM_STALL_TIMEOUT_MS * 2);
    expect(handleReconnect).not.toHaveBeenCalled();
  });

  it("resets timer when activityToken changes", () => {
    const { handleReconnect, args } = setup({ activityToken: "1:0" });
    const { rerender } = renderHook(
      (props: typeof args) => useStreamActivityWatchdog(props),
      { initialProps: args },
    );
    vi.advanceTimersByTime(STREAM_STALL_TIMEOUT_MS - 5_000);
    rerender({ ...args, activityToken: "1:1" });
    vi.advanceTimersByTime(STREAM_STALL_TIMEOUT_MS - 5_000);
    expect(handleReconnect).not.toHaveBeenCalled();
    vi.advanceTimersByTime(5_000);
    expect(handleReconnect).toHaveBeenCalledTimes(1);
  });

  it("clears timer on unmount", () => {
    const { handleReconnect, args } = setup();
    const { unmount } = renderHook(() => useStreamActivityWatchdog(args));
    vi.advanceTimersByTime(STREAM_STALL_TIMEOUT_MS - 1_000);
    unmount();
    vi.advanceTimersByTime(STREAM_STALL_TIMEOUT_MS);
    expect(handleReconnect).not.toHaveBeenCalled();
  });
});
