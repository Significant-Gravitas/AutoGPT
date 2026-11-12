import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/actions", () => ({ getWebSocketToken: vi.fn() }));

import { getWebSocketToken } from "@/lib/auth/actions";
import BackendAPI from "../client";
import type { GraphExecutionID } from "../types";
import { FakeWebSocket } from "./websocket-fake";

describe("WebSocket send deadlines", () => {
  let api: BackendAPI;
  const expiredID = "00000000-0000-4000-8000-000000000001" as GraphExecutionID;
  const nextID = "00000000-0000-4000-8000-000000000002" as GraphExecutionID;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", FakeWebSocket);
    FakeWebSocket.instances = [];
    vi.mocked(getWebSocketToken)
      .mockReset()
      .mockResolvedValue({ token: "test-token" });
    api = new BackendAPI("http://test", "ws://test/ws");
  });

  afterEach(() => {
    api.disconnectWebSocket();
    vi.clearAllTimers();
    vi.unstubAllGlobals();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it.each(["returned", "thrown"])(
    "times out a send during persistent %s token failures while allowing later recovery",
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
      api.onWebSocketConnect(connected);
      const expired = vi.fn();
      const first = api.subscribeToGraphExecution(expiredID).catch(expired);

      await vi.advanceTimersByTimeAsync(15_000);
      const next = api.subscribeToGraphExecution(nextID);
      await vi.advanceTimersByTimeAsync(14_999);
      expect(expired).not.toHaveBeenCalled();
      expect(vi.mocked(getWebSocketToken).mock.calls.length).toBeGreaterThan(1);
      await vi.advanceTimersByTimeAsync(1);
      expect(expired).toHaveBeenCalledExactlyOnceWith(
        new Error("WebSocket connection timed out"),
      );
      await first;

      vi.mocked(getWebSocketToken).mockResolvedValue({
        token: "recovered-token",
      });
      await vi.advanceTimersByTimeAsync(1000);
      const socket = FakeWebSocket.instances[0];
      socket.open();
      socket.pong();
      await next;
      expect(connected).toHaveBeenCalledTimes(1);
      expect(
        socket.send.mock.calls.map(([message]) => JSON.parse(message)),
      ).toEqual([
        { method: "heartbeat", data: "ping", success: true },
        {
          method: "subscribe_graph_execution",
          data: { graph_exec_id: nextID },
        },
      ]);
      api.disconnectWebSocket();
      expect(vi.getTimerCount()).toBe(0);
    },
  );

  it("times out while the token request is stalled and never sends the expired message", async () => {
    let resolveToken: ((value: { token: string }) => void) | undefined;
    vi.mocked(getWebSocketToken).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveToken = resolve;
        }),
    );
    const rejected = vi.fn();
    const pending = api.subscribeToGraphExecution(expiredID).catch(rejected);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(rejected).toHaveBeenCalledExactlyOnceWith(
      new Error("WebSocket connection timed out"),
    );
    await pending;

    resolveToken?.({ token: "recovered-token" });
    await vi.advanceTimersByTimeAsync(0);
    const socket = FakeWebSocket.instances[0];
    socket.open();
    socket.pong();
    await vi.advanceTimersByTimeAsync(0);
    expect(socket.send).toHaveBeenCalledExactlyOnceWith(
      JSON.stringify({ method: "heartbeat", data: "ping", success: true }),
    );
    api.disconnectWebSocket();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("times out when the socket never opens", async () => {
    const rejected = vi.fn();
    const pending = api.subscribeToGraphExecution(expiredID).catch(rejected);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(rejected).toHaveBeenCalledExactlyOnceWith(
      new Error("WebSocket connection timed out"),
    );
    await pending;
    const socket = FakeWebSocket.instances[0];
    socket.open();
    socket.pong();
    await vi.advanceTimersByTimeAsync(0);
    expect(socket.send).toHaveBeenCalledExactlyOnceWith(
      JSON.stringify({ method: "heartbeat", data: "ping", success: true }),
    );
  });

  it("clears the send deadline after authentication succeeds", async () => {
    const pending = api.subscribeToGraphExecution(nextID);
    await vi.advanceTimersByTimeAsync(0);
    const socket = FakeWebSocket.instances[0];
    socket.open();
    socket.pong();
    await pending;
    api.disconnectWebSocket();
    expect(vi.getTimerCount()).toBe(0);
  });
});
