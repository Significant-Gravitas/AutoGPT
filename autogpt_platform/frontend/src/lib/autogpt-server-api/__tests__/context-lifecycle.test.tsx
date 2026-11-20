import { act, cleanup, render } from "@testing-library/react";
import { StrictMode, useEffect } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/actions", () => ({ getWebSocketToken: vi.fn() }));

import { getWebSocketToken } from "@/lib/auth/actions";
import { BackendAPIProvider, useBackendAPI } from "../context";
import { FakeWebSocket } from "./websocket-fake";

interface Props {
  onConnected: () => void;
}

function Subscriber({ onConnected }: Props) {
  const api = useBackendAPI();
  useEffect(() => api.onWebSocketConnect(onConnected), [api, onConnected]);
  return null;
}

describe("BackendAPIProvider connection lifetime", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", FakeWebSocket);
    FakeWebSocket.instances = [];
    vi.mocked(getWebSocketToken)
      .mockReset()
      .mockResolvedValue({ token: "test-token" });
  });

  afterEach(() => {
    cleanup();
    vi.clearAllTimers();
    vi.unstubAllGlobals();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it.each(["returned", "thrown"])(
    "stops %s token failures from retrying after unmount",
    async (failure) => {
      if (failure === "returned") {
        vi.mocked(getWebSocketToken).mockResolvedValue({
          token: null,
          error: "Authentication service unavailable",
          retryable: true,
        });
      } else {
        vi.mocked(getWebSocketToken).mockRejectedValue(
          new TypeError("Network error"),
        );
      }
      const connected = vi.fn();
      const view = render(
        <BackendAPIProvider>
          <Subscriber onConnected={connected} />
        </BackendAPIProvider>,
      );
      await act(() => vi.advanceTimersByTimeAsync(1000));
      const attempts = vi.mocked(getWebSocketToken).mock.calls.length;
      expect(attempts).toBeGreaterThan(1);

      view.unmount();
      await act(() => vi.advanceTimersByTimeAsync(120_000));
      expect(getWebSocketToken).toHaveBeenCalledTimes(attempts);
      expect(connected).not.toHaveBeenCalled();
      expect(vi.getTimerCount()).toBe(0);
    },
  );

  it.each(["resolved", "rejected"])(
    "ignores a token request %s after unmount",
    async (result) => {
      let resolveToken: ((value: { token: string }) => void) | undefined;
      let rejectToken: ((error: Error) => void) | undefined;
      vi.mocked(getWebSocketToken).mockImplementationOnce(
        () =>
          new Promise((resolve, reject) => {
            resolveToken = resolve;
            rejectToken = reject;
          }),
      );
      const connected = vi.fn();
      const view = render(
        <BackendAPIProvider>
          <Subscriber onConnected={connected} />
        </BackendAPIProvider>,
      );
      expect(getWebSocketToken).toHaveBeenCalledTimes(1);
      view.unmount();

      if (result === "resolved") resolveToken?.({ token: "late-token" });
      else rejectToken?.(new TypeError("Network error"));
      await act(() => vi.advanceTimersByTimeAsync(120_000));
      expect(getWebSocketToken).toHaveBeenCalledTimes(1);
      expect(FakeWebSocket.instances).toHaveLength(0);
      expect(connected).not.toHaveBeenCalled();
      expect(vi.getTimerCount()).toBe(0);
    },
  );

  it.each([false, true])(
    "authenticates and closes its socket on unmount (Strict Mode: %s)",
    async (strict) => {
      const connected = vi.fn();
      const provider = (
        <BackendAPIProvider>
          <Subscriber onConnected={connected} />
        </BackendAPIProvider>
      );
      const view = render(
        strict ? <StrictMode>{provider}</StrictMode> : provider,
      );
      await act(() => vi.advanceTimersByTimeAsync(0));
      expect(FakeWebSocket.instances).toHaveLength(1);
      const socket = FakeWebSocket.instances[0];
      act(() => {
        socket.open();
        socket.pong();
      });
      expect(connected).toHaveBeenCalledTimes(1);

      view.unmount();
      expect(socket.close).toHaveBeenCalledTimes(1);
      await act(() => vi.advanceTimersByTimeAsync(120_000));
      expect(FakeWebSocket.instances).toHaveLength(1);
      expect(vi.getTimerCount()).toBe(0);
    },
  );
});
