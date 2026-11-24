import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/actions", () => ({ getWebSocketToken: vi.fn() }));

import { getWebSocketToken } from "@/lib/auth/actions";
import {
  clearWebSocketDisconnectIntent,
  hasWebSocketDisconnectIntent,
  setWebSocketDisconnectIntent,
} from "@/lib/auth/helpers";
import BackendAPI from "../client";
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

  it("backs off transient auth failures and resets only after authentication", async () => {
    void api.connectWebSocket();
    const first = await socketAt();
    first.open();
    first.serverClose(1013);
    await vi.advanceTimersByTimeAsync(999);
    expect(FakeWebSocket.instances).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    const second = await socketAt(1);
    second.open();
    second.serverClose(1013);
    await vi.advanceTimersByTimeAsync(1999);
    expect(FakeWebSocket.instances).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(1);
    const third = await socketAt(2);
    third.open();
    third.pong();
    third.serverClose(1006);
    await vi.advanceTimersByTimeAsync(1000);
    expect(FakeWebSocket.instances).toHaveLength(4);
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
    const delays = [1000, 2000, 4000, 8000, 16000, 30000, 30000];
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
