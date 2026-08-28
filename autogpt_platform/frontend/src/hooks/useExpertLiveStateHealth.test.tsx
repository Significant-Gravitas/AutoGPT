import { act, render, renderHook, screen } from "@testing-library/react";
import * as Sentry from "@sentry/nextjs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LiveUpdatesStatus } from "@/components/molecules/LiveUpdatesStatus/LiveUpdatesStatus";
import {
  liveStateHealthTestUtils,
  useExpertLiveStateHealth,
} from "./useExpertLiveStateHealth";

const api = vi.hoisted(() => {
  let connectHandler: (() => void) | null = null;
  let disconnectHandler: (() => void) | null = null;
  let notificationHandler: (() => void) | null = null;
  return {
    connectWebSocket: vi.fn().mockResolvedValue(undefined),
    onWebSocketConnect: vi.fn((handler: () => void) => {
      connectHandler = handler;
      return vi.fn();
    }),
    onWebSocketDisconnect: vi.fn((handler: () => void) => {
      disconnectHandler = handler;
      return vi.fn();
    }),
    onWebSocketMessage: vi.fn((method: string, handler: () => void) => {
      if (method === "notification") notificationHandler = handler;
      return vi.fn();
    }),
    connect() {
      connectHandler?.();
    },
    disconnect() {
      disconnectHandler?.();
    },
    notify() {
      notificationHandler?.();
    },
    reset() {
      connectHandler = null;
      disconnectHandler = null;
      notificationHandler = null;
      this.connectWebSocket.mockClear();
      this.onWebSocketConnect.mockClear();
      this.onWebSocketDisconnect.mockClear();
      this.onWebSocketMessage.mockClear();
    },
  };
});

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => api,
}));
vi.mock("@sentry/nextjs", () => ({ captureMessage: vi.fn() }));

beforeEach(() => {
  vi.useFakeTimers();
  api.reset();
  liveStateHealthTestUtils.reset();
  vi.mocked(Sentry.captureMessage).mockReset();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("useExpertLiveStateHealth", () => {
  it("falls back to bounded polling and returns to live updates", () => {
    const refresh = vi.fn().mockResolvedValue(undefined);
    const { result } = renderHook(() =>
      useExpertLiveStateHealth({
        surface: "home",
        hireExpertsEnabled: true,
        onFallbackRefresh: refresh,
      }),
    );

    expect(result.current).toBe("connecting");
    act(() => vi.advanceTimersByTime(4_000));
    expect(result.current).toBe("polling");
    expect(refresh).toHaveBeenCalledTimes(1);

    act(() => vi.advanceTimersByTime(10_000));
    expect(refresh).toHaveBeenCalledTimes(3);
    expect(Sentry.captureMessage).toHaveBeenCalledWith(
      "expert_live_state_polling_fallback",
      expect.objectContaining({
        tags: expect.objectContaining({ surface: "home" }),
      }),
    );

    act(() => api.connect());
    expect(result.current).toBe("live");
    act(() => api.notify());
    expect(refresh).toHaveBeenCalledTimes(5);
    act(() => vi.advanceTimersByTime(10_000));
    expect(refresh).toHaveBeenCalledTimes(5);
  });

  it("records a cross-surface hire-experts mismatch", () => {
    const home = renderHook(() =>
      useExpertLiveStateHealth({
        surface: "home",
        hireExpertsEnabled: true,
      }),
    );
    const team = renderHook(() =>
      useExpertLiveStateHealth({
        surface: "team",
        hireExpertsEnabled: false,
      }),
    );

    expect(Sentry.captureMessage).toHaveBeenCalledWith(
      "hire_experts_surface_flag_mismatch",
      expect.objectContaining({ level: "error" }),
    );
    home.unmount();
    team.unmount();
  });
});

describe("LiveUpdatesStatus", () => {
  it("explains the automatic polling fallback without technical detail", () => {
    const { rerender } = render(<LiveUpdatesStatus health="live" />);
    expect(screen.queryByRole("status")).toBeNull();

    rerender(<LiveUpdatesStatus health="polling" />);
    const status = screen.getByRole("status");
    expect(status.textContent).toContain(
      "Progress is refreshing automatically",
    );
    expect(status.textContent).not.toMatch(/websocket/i);
  });
});
