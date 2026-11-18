import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/actions", () => ({ getWebSocketToken: vi.fn() }));

import { getWebSocketToken } from "@/lib/auth/actions";
import {
  clearWebSocketDisconnectIntent,
  hasWebSocketDisconnectIntent,
  setWebSocketDisconnectIntent,
} from "@/lib/auth/helpers";
import BackendAPI from "../client";
import type { GraphID } from "../types";
import { FakeWebSocket } from "./websocket-fake";

describe("WebSocket recovery and logout", () => {
  let api: BackendAPI;
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", FakeWebSocket);
    FakeWebSocket.instances = [];
    vi.mocked(getWebSocketToken)
      .mockReset()
      .mockResolvedValue({ token: "test-token" });
    // The low end of each jittered retry window: half the backoff step.
    vi.spyOn(Math, "random").mockReturnValue(0);
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

  it("keeps logout intent when the transport opens late", async () => {
    const connected = vi.fn();
    api.onWebSocketConnect(connected);
    const socket = await socketAt();
    setWebSocketDisconnectIntent();
    socket.open();
    socket.pong();
    expect(hasWebSocketDisconnectIntent()).toBe(true);
    expect(connected).not.toHaveBeenCalled();
    expect(socket.close).toHaveBeenCalled();
  });

  it("does not create a socket when logout happens during token retrieval", async () => {
    let finishToken: ((value: { token: string }) => void) | undefined;
    vi.mocked(getWebSocketToken).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finishToken = resolve;
        }),
    );
    const pending = api.connectWebSocket();
    api.disconnectWebSocket();
    finishToken?.({ token: "stale" });
    await Promise.resolve();
    await Promise.resolve();
    expect(FakeWebSocket.instances).toHaveLength(0);
    await pending;
  });

  it("keeps backing off until a connection has stayed up for 10 s", async () => {
    void api.connectWebSocket();
    const first = await socketAt();
    first.open();
    first.serverClose(1013);
    await vi.advanceTimersByTimeAsync(499);
    expect(FakeWebSocket.instances).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    const second = await socketAt(1);
    second.open();
    second.serverClose(1013);
    await vi.advanceTimersByTimeAsync(1000);
    // Authenticated, then dropped at once, as while the server's event
    // stream keeps failing: that is no reason to retry quickly again.
    const third = await socketAt(2);
    third.open();
    third.pong();
    third.serverClose(1011);
    await vi.advanceTimersByTimeAsync(1999);
    expect(FakeWebSocket.instances).toHaveLength(3);
    await vi.advanceTimersByTimeAsync(1);
    const fourth = await socketAt(3);
    fourth.open();
    fourth.pong();
    // A reconnect counts as healthy from the answer to its ordering ping.
    await vi.advanceTimersByTimeAsync(0);
    fourth.pong();
    await vi.advanceTimersByTimeAsync(10_000);
    fourth.serverClose(1006);
    await vi.advanceTimersByTimeAsync(500);
    expect(FakeWebSocket.instances).toHaveLength(5);
  });

  it("spreads reconnect attempts with jitter", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    void api.connectWebSocket();
    const first = await socketAt();
    first.open();
    first.serverClose(1013);
    await vi.advanceTimersByTimeAsync(749);
    expect(FakeWebSocket.instances).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it("reconnects at once when the browser comes back online", async () => {
    void api.connectWebSocket();
    const first = await socketAt();
    first.open();
    first.pong();
    first.serverClose(1006);
    await vi.advanceTimersByTimeAsync(100);
    expect(FakeWebSocket.instances).toHaveLength(1);

    window.dispatchEvent(new Event("online"));
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeWebSocket.instances).toHaveLength(2);
    // The retry it replaced does not fire as well.
    await vi.advanceTimersByTimeAsync(1000);
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it("replaces a connection that stops answering once the tab is back in view", async () => {
    void api.connectWebSocket();
    const socket = await socketAt();
    socket.open();
    socket.pong();
    socket.send.mockClear();

    document.dispatchEvent(new Event("visibilitychange"));
    expect(socket.send).toHaveBeenCalledWith(
      expect.stringContaining('"heartbeat"'),
    );
    // A second check while the first is out neither pings again nor pushes
    // its deadline back.
    await vi.advanceTimersByTimeAsync(5000);
    window.dispatchEvent(new Event("focus"));
    expect(socket.send).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(5000);
    expect(socket.close).toHaveBeenCalled();
    socket.serverClose(1006);
    await vi.advanceTimersByTimeAsync(500);
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it("starts the backoff over for the next session after a disconnect", async () => {
    void api.connectWebSocket();
    const first = await socketAt();
    first.open();
    first.serverClose(1013);
    await vi.advanceTimersByTimeAsync(500);
    const second = await socketAt(1);
    second.open();
    second.serverClose(1013);
    // Signed out, then in again.
    api.disconnectWebSocket();
    void api.connectWebSocket();
    const third = await socketAt(2);
    third.open();
    third.serverClose(1013);
    await vi.advanceTimersByTimeAsync(499);
    expect(FakeWebSocket.instances).toHaveLength(3);
    await vi.advanceTimersByTimeAsync(1);
    expect(FakeWebSocket.instances).toHaveLength(4);
  });

  it("checks an open connection when the window regains focus", async () => {
    void api.connectWebSocket();
    const socket = await socketAt();
    socket.open();
    socket.pong();
    socket.send.mockClear();

    window.dispatchEvent(new Event("focus"));

    expect(socket.send).toHaveBeenCalledTimes(1);
    expect(socket.send).toHaveBeenCalledWith(
      expect.stringContaining('"heartbeat"'),
    );
  });

  it("gives the resubscriptions 30 s to be answered, then replaces the socket", async () => {
    const reconnected = vi.fn();
    api.onWebSocketReconnect(reconnected);
    void api.connectWebSocket();
    const first = await socketAt();
    first.open();
    first.pong();
    first.serverClose(1011);
    await vi.advanceTimersByTimeAsync(500);
    const second = await socketAt(1);
    second.open();
    second.pong();

    // Longer than a heartbeat's 10 s: opening a shard's event stream can
    // take the server up to 15 s.
    await vi.advanceTimersByTimeAsync(29_999);
    expect(second.close).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(second.close).toHaveBeenCalledTimes(1);
    expect(reconnected).not.toHaveBeenCalled();
  });

  it("stays disconnected when the page comes back after a disconnect", async () => {
    void api.connectWebSocket();
    const socket = await socketAt();
    socket.open();
    socket.pong();
    api.disconnectWebSocket();

    window.dispatchEvent(new Event("online"));
    document.dispatchEvent(new Event("visibilitychange"));
    await vi.advanceTimersByTimeAsync(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("calls reconnect handlers once the server has handled the resubscriptions", async () => {
    const reconnected = vi.fn();
    api.onWebSocketConnect(
      () => void api.subscribeToGraphExecutions("graph-1" as GraphID),
    );
    api.onWebSocketReconnect(reconnected);
    const first = await socketAt();
    first.open();
    first.pong();
    expect(reconnected).not.toHaveBeenCalled();

    first.serverClose(1011);
    await vi.advanceTimersByTimeAsync(500);
    const second = await socketAt(1);
    second.open();
    second.pong();
    await vi.advanceTimersByTimeAsync(0);
    // The server answers a socket's messages in order, so the pong to a
    // ping sent after the resubscription means the server has handled it.
    expect(
      second.send.mock.calls.map(([data]) => JSON.parse(data).method),
    ).toEqual(["heartbeat", "subscribe_graph_executions", "heartbeat"]);
    expect(reconnected).not.toHaveBeenCalled();
    second.pong();
    expect(reconnected).toHaveBeenCalledTimes(1);
  });

  it("sends the ordering ping after what a view subscribes during the spread", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    api.onWebSocketConnect(
      () => void api.subscribeToGraphExecutions("graph-1" as GraphID),
    );
    const first = await socketAt();
    first.open();
    first.pong();
    first.serverClose(1011);
    await vi.advanceTimersByTimeAsync(750);
    const second = await socketAt(1);
    second.open();
    second.pong();
    await vi.advanceTimersByTimeAsync(500);
    void api.subscribeToGraphExecutions("graph-2" as GraphID);
    await vi.advanceTimersByTimeAsync(500);

    expect(
      second.send.mock.calls.map(([data]) => JSON.parse(data).method),
    ).toEqual([
      "heartbeat",
      "subscribe_graph_executions",
      "subscribe_graph_executions",
      "heartbeat",
    ]);
  });

  it("asks again when a view subscribes while the ordering ping is out", async () => {
    const reconnected = vi.fn();
    api.onWebSocketReconnect(reconnected);
    void api.connectWebSocket();
    const first = await socketAt();
    first.open();
    first.pong();
    first.serverClose(1011);
    await vi.advanceTimersByTimeAsync(500);
    const second = await socketAt(1);
    second.open();
    second.pong();
    await vi.advanceTimersByTimeAsync(0);
    void api.subscribeToGraphExecutions("graph-2" as GraphID);

    // That pong can't cover graph-2, sent after its ping.
    second.pong();
    expect(reconnected).not.toHaveBeenCalled();
    expect(
      second.send.mock.calls.map(([data]) => JSON.parse(data).method),
    ).toEqual([
      "heartbeat",
      "heartbeat",
      "subscribe_graph_executions",
      "heartbeat",
    ]);
    second.pong();
    expect(reconnected).toHaveBeenCalledTimes(1);
  });

  it("keeps waiting while the server is still answering the resubscriptions", async () => {
    void api.connectWebSocket();
    const first = await socketAt();
    first.open();
    first.pong();
    first.serverClose(1011);
    await vi.advanceTimersByTimeAsync(500);
    const second = await socketAt(1);
    second.open();
    second.pong();
    await vi.advanceTimersByTimeAsync(25_000);
    second.onmessage?.(
      new MessageEvent("message", {
        data: JSON.stringify({
          method: "subscribe_graph_executions",
          success: true,
        }),
      }),
    );

    await vi.advanceTimersByTimeAsync(29_999);
    expect(second.close).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(second.close).toHaveBeenCalledTimes(1);
  });

  it("counts a reconnect as healthy only once it has resubscribed", async () => {
    void api.connectWebSocket();
    const first = await socketAt();
    first.open();
    first.pong();
    await vi.advanceTimersByTimeAsync(10_000);
    first.serverClose(1011);
    await vi.advanceTimersByTimeAsync(500);
    const second = await socketAt(1);
    second.open();
    second.pong();
    // The ordering ping is never answered.
    await vi.advanceTimersByTimeAsync(30_000);
    expect(second.close).toHaveBeenCalled();
    second.serverClose(1006);

    await vi.advanceTimersByTimeAsync(999);
    expect(FakeWebSocket.instances).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(1);
    expect(FakeWebSocket.instances).toHaveLength(3);
  });

  it("keeps the periodic heartbeat from pushing back a check that is out", async () => {
    void api.connectWebSocket();
    const socket = await socketAt();
    socket.open();
    socket.pong();
    await vi.advanceTimersByTimeAsync(91_000);
    window.dispatchEvent(new Event("focus"));

    await vi.advanceTimersByTimeAsync(9_999);
    expect(socket.close).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(socket.close).toHaveBeenCalledTimes(1);
  });

  it("spreads the refetches after a reconnect over up to 2 s", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const reconnected = vi.fn();
    api.onWebSocketReconnect(reconnected);
    void api.connectWebSocket();
    const first = await socketAt();
    first.open();
    first.pong();
    first.serverClose(1011);
    await vi.advanceTimersByTimeAsync(750);
    const second = await socketAt(1);
    second.open();
    second.pong();

    // The ordering ping, and so the refetch, waits out the spread.
    await vi.advanceTimersByTimeAsync(999);
    expect(second.send).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(second.send).toHaveBeenCalledTimes(2);
    expect(reconnected).not.toHaveBeenCalled();
    second.pong();
    expect(reconnected).toHaveBeenCalledTimes(1);
  });

  it("does not count the first connection after a disconnect as a reconnect", async () => {
    const reconnected = vi.fn();
    api.onWebSocketReconnect(reconnected);
    void api.connectWebSocket();
    const first = await socketAt();
    first.open();
    first.pong();
    // Signed out, then in again.
    api.disconnectWebSocket();
    void api.connectWebSocket();
    const second = await socketAt(1);
    second.open();
    second.pong();

    await vi.advanceTimersByTimeAsync(5000);
    expect(second.send).toHaveBeenCalledTimes(1);
    expect(reconnected).not.toHaveBeenCalled();
  });

  it("ignores callbacks from a socket replaced after disconnect", async () => {
    const connected = vi.fn();
    api.onWebSocketConnect(connected);
    const old = await socketAt();
    old.open();
    old.pong();
    old.beginClose();
    api.disconnectWebSocket();
    const reconnecting = api.connectWebSocket();
    const current = await socketAt(1);
    current.open();
    current.pong();
    await reconnecting;
    old.serverClose(1006);
    old.pong();
    await vi.advanceTimersByTimeAsync(1000);
    expect(FakeWebSocket.instances).toHaveLength(2);
    expect(connected).toHaveBeenCalledTimes(2);
    api.disconnectWebSocket();
    expect(vi.getTimerCount()).toBe(0);
  });
  it("closes a provisional connection that never answers the authentication ping", async () => {
    const connected = vi.fn();
    api.onWebSocketConnect(connected);
    const socket = await socketAt();
    socket.open();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(socket.close).toHaveBeenCalledTimes(1);
    expect(connected).not.toHaveBeenCalled();
    socket.serverClose(1006);
    await vi.advanceTimersByTimeAsync(1000);
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it("bounds exponential retries and cancels the pending retry on disconnect", async () => {
    void api.connectWebSocket();
    const delays = [500, 1000, 2000, 4000, 8000, 15000, 15000];
    for (const [index, delay] of delays.entries()) {
      const socket = await socketAt(index);
      socket.open();
      socket.serverClose(1013);
      await vi.advanceTimersByTimeAsync(delay - 1);
      expect(FakeWebSocket.instances).toHaveLength(index + 1);
      await vi.advanceTimersByTimeAsync(1);
    }
    const last = await socketAt(delays.length);
    last.serverClose(1013);
    api.disconnectWebSocket();
    await vi.advanceTimersByTimeAsync(60_000);
    expect(FakeWebSocket.instances).toHaveLength(delays.length + 1);
  });
  it("can retry token retrieval after explicitly disconnecting an old socket", async () => {
    const connected = api.connectWebSocket();
    const old = await socketAt();
    old.open();
    old.pong();
    await connected;
    api.disconnectWebSocket();
    vi.mocked(getWebSocketToken).mockResolvedValueOnce({ token: null });
    await api.connectWebSocket();
    const reconnecting = api.connectWebSocket();
    const current = await socketAt(1);
    current.open();
    current.pong();
    await reconnecting;
    old.serverClose(1006);
    expect(FakeWebSocket.instances).toHaveLength(2);
    api.disconnectWebSocket();
    expect(vi.getTimerCount()).toBe(0);
  });
});
