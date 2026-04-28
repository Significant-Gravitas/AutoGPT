import { renderHook } from "@testing-library/react";
import { act } from "react";
import { useRef } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useStreamActivityWatchdog } from "../useStreamActivityWatchdog";

type Status = "submitted" | "streaming" | "ready" | "error";

type HookArgs = Parameters<typeof useStreamActivityWatchdog>[0];

function setup(initial: Partial<HookArgs> = {}) {
  const reconnect = vi.fn();
  const { result, rerender } = renderHook(
    ({
      status,
      activityToken,
      isReconnectScheduled,
      isUserStopping,
      sessionId,
    }: {
      status: Status;
      activityToken: string | number;
      isReconnectScheduled: boolean;
      isUserStopping: boolean;
      sessionId: string | null;
    }) => {
      const isUserStoppingRef = useRef(isUserStopping);
      isUserStoppingRef.current = isUserStopping;
      const handleReconnectRef = useRef(reconnect);
      handleReconnectRef.current = reconnect;
      useStreamActivityWatchdog({
        sessionId,
        status,
        activityToken,
        isReconnectScheduled,
        isUserStoppingRef,
        handleReconnectRef,
      });
      return { reconnect };
    },
    {
      initialProps: {
        status: "streaming" as Status,
        activityToken: initial.activityToken ?? "0",
        isReconnectScheduled: initial.isReconnectScheduled ?? false,
        isUserStopping: false,
        sessionId:
          initial.sessionId === undefined ? "sess-1" : initial.sessionId,
      },
    },
  );
  return { result, rerender, reconnect };
}

describe("useStreamActivityWatchdog", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("fires reconnect after 60 s of inactivity while streaming", () => {
    const { reconnect } = setup();

    act(() => {
      vi.advanceTimersByTime(59_000);
    });
    expect(reconnect).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(1_000);
    });
    expect(reconnect).toHaveBeenCalledTimes(1);
  });

  it("resets the timer on activity", () => {
    const { rerender, reconnect } = setup({ activityToken: "0" });

    act(() => {
      vi.advanceTimersByTime(50_000);
    });

    rerender({
      status: "streaming",
      activityToken: "1",
      isReconnectScheduled: false,
      isUserStopping: false,
      sessionId: "sess-1",
    });

    act(() => {
      vi.advanceTimersByTime(59_000);
    });
    expect(reconnect).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(1_000);
    });
    expect(reconnect).toHaveBeenCalledTimes(1);
  });

  it("does not fire when a reconnect is already scheduled", () => {
    const { reconnect } = setup({ isReconnectScheduled: true });

    act(() => {
      vi.advanceTimersByTime(120_000);
    });
    expect(reconnect).not.toHaveBeenCalled();
  });

  it("does not fire for ready / error status", () => {
    const { rerender, reconnect } = setup();

    rerender({
      status: "ready",
      activityToken: "0",
      isReconnectScheduled: false,
      isUserStopping: false,
      sessionId: "sess-1",
    });

    act(() => {
      vi.advanceTimersByTime(120_000);
    });
    expect(reconnect).not.toHaveBeenCalled();
  });

  it("does not fire when the user explicitly stopped", () => {
    const { rerender, reconnect } = setup();

    rerender({
      status: "streaming",
      activityToken: "0",
      isReconnectScheduled: false,
      isUserStopping: true,
      sessionId: "sess-1",
    });

    act(() => {
      vi.advanceTimersByTime(120_000);
    });
    expect(reconnect).not.toHaveBeenCalled();
  });

  it("does not fire without a session id", () => {
    const { rerender, reconnect } = setup();

    rerender({
      status: "streaming",
      activityToken: "0",
      isReconnectScheduled: false,
      isUserStopping: false,
      sessionId: null,
    });

    act(() => {
      vi.advanceTimersByTime(120_000);
    });
    expect(reconnect).not.toHaveBeenCalled();
  });
});
