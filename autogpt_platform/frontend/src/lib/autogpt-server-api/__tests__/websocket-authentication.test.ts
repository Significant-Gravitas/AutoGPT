import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/actions", () => ({ getWebSocketToken: vi.fn() }));

import { getWebSocketToken } from "@/lib/auth/actions";
import {
  clearWebSocketDisconnectIntent,
  setWebSocketDisconnectIntent,
} from "@/lib/auth/helpers";
import BackendAPI from "../client";
import type { GraphExecutionID } from "../types";
import { FakeWebSocket } from "./websocket-fake";

describe("WebSocket authentication and recovery", () => {
  let api: BackendAPI;
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
    clearWebSocketDisconnectIntent();
    vi.clearAllTimers();
    vi.unstubAllGlobals();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });
  async function socketAt(index = 0) {
    await vi.waitFor(
      () => expect(FakeWebSocket.instances.length).toBeGreaterThan(index),
      { interval: 1 },
    );
    return FakeWebSocket.instances[index];
  }

  it.each([4001, 4002, 4003])(
    "does not report authenticated connection or retry after close %s",
    async (code) => {
      const connected = vi.fn();
      const disconnected = vi.fn();
      api.onWebSocketConnect(connected);
      api.onWebSocketDisconnect(disconnected);
      const socket = await socketAt();
      socket.open();
      expect(connected).not.toHaveBeenCalled();
      const lateSubscriber = vi.fn();
      api.onWebSocketConnect(lateSubscriber);
      expect(lateSubscriber).not.toHaveBeenCalled();
      socket.serverClose(code);
      await vi.advanceTimersByTimeAsync(60_000);
      expect(FakeWebSocket.instances).toHaveLength(1);
      expect(disconnected).not.toHaveBeenCalled();
    },
  );

  it("waits for a pong before subscriptions and connects exactly once", async () => {
    const connected = vi.fn();
    api.onWebSocketConnect(connected);
    const executionID =
      "00000000-0000-4000-8000-000000000001" as GraphExecutionID;
    const subscription = api.subscribeToGraphExecution(executionID);
    const socket = await socketAt();
    socket.open();
    expect(
      socket.send.mock.calls.map(([data]) => JSON.parse(data).method),
    ).toEqual(["heartbeat"]);
    expect(connected).not.toHaveBeenCalled();
    socket.pong();
    await subscription;
    socket.pong();
    expect(connected).toHaveBeenCalledTimes(1);
    expect(socket.send).toHaveBeenLastCalledWith(
      JSON.stringify({
        method: "subscribe_graph_execution",
        data: { graph_exec_id: executionID },
      }),
    );
  });

  it.each([{ token: null }, { token: null, error: "unavailable" }])(
    "settles without connecting when no token is issued: %s",
    async (result) => {
      vi.mocked(getWebSocketToken).mockResolvedValueOnce(result);
      await api.connectWebSocket();
      expect(FakeWebSocket.instances).toHaveLength(0);
      const next = api.connectWebSocket();
      const socket = await socketAt();
      socket.open();
      socket.pong();
      await next;
    },
  );

  it("settles a failed token request without opening a socket", async () => {
    vi.mocked(getWebSocketToken).mockRejectedValueOnce(
      new Error("unavailable"),
    );
    await api.connectWebSocket();
    expect(FakeWebSocket.instances).toHaveLength(0);
  });

  it("does not fetch a token while logout intent is present", async () => {
    setWebSocketDisconnectIntent();
    await api.connectWebSocket();
    expect(getWebSocketToken).not.toHaveBeenCalled();
  });

  it("keeps heartbeats and incoming events working after authentication", async () => {
    const connected = vi.fn();
    const notification = vi.fn();
    api.onWebSocketConnect(connected);
    api.onWebSocketMessage("notification", notification);
    const socket = await socketAt();
    socket.open();
    socket.pong();
    socket.onmessage?.(
      new MessageEvent("message", {
        data: JSON.stringify({
          method: "notification",
          data: { id: "notice" },
        }),
      }),
    );
    expect(notification).toHaveBeenCalledWith({ id: "notice" });
    await vi.advanceTimersByTimeAsync(100_000);
    expect(JSON.parse(socket.send.mock.lastCall![0])).toMatchObject({
      method: "heartbeat",
      data: "ping",
    });
    socket.pong();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(socket.close).not.toHaveBeenCalled();
    expect(connected).toHaveBeenCalledTimes(1);
  });
});
