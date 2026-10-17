import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/actions", () => ({
  getWebSocketToken: vi.fn(async () => ({ token: "test-token" })),
}));

import BackendAPI from "../client";
import type { GraphExecutionID } from "../types";
import { FakeWebSocket } from "./websocket-fake";

describe("pending subscriptions share connection retries", () => {
  let api: BackendAPI;
  const executionID =
    "00000000-0000-4000-8000-000000000001" as GraphExecutionID;
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", FakeWebSocket);
    FakeWebSocket.instances = [];
    api = new BackendAPI("http://test", "ws://test/ws");
  });
  afterEach(() => {
    api.disconnectWebSocket();
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
    "rejects the pending send without retrying auth close %s",
    async (code) => {
      const outcome = api.subscribeToGraphExecution(executionID).then(
        () => "sent",
        () => "rejected",
      );
      const socket = await socketAt();
      socket.open();
      socket.serverClose(code);
      await vi.advanceTimersByTimeAsync(60_000);
      expect(FakeWebSocket.instances).toHaveLength(1);
      expect(await outcome).toBe("rejected");
    },
  );

  it("waits for the shared backoff and authenticated pong before sending", async () => {
    const sent = vi.fn();
    const outcome = api.subscribeToGraphExecution(executionID).then(sent);
    const first = await socketAt();
    first.open();
    first.serverClose(1013);
    await vi.advanceTimersByTimeAsync(999);
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(sent).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    const next = await socketAt(1);
    next.open();
    next.pong();
    await outcome;
    expect(sent).toHaveBeenCalledTimes(1);
    expect(next.send).toHaveBeenLastCalledWith(
      JSON.stringify({
        method: "subscribe_graph_execution",
        data: { graph_exec_id: executionID },
      }),
    );
  });

  it.each([4001, 4002, 4003])(
    "waits for a closing socket's auth code %s before rejecting a new send",
    async (code) => {
      const connected = api.connectWebSocket();
      const socket = await socketAt();
      socket.open();
      socket.pong();
      await connected;
      socket.beginClose();
      const outcome = api.subscribeToGraphExecution(executionID).then(
        () => "sent",
        () => "rejected",
      );
      await vi.advanceTimersByTimeAsync(0);
      expect(FakeWebSocket.instances).toHaveLength(1);
      socket.serverClose(code);
      expect(await outcome).toBe("rejected");
      await vi.advanceTimersByTimeAsync(60_000);
      expect(FakeWebSocket.instances).toHaveLength(1);
      expect(vi.getTimerCount()).toBe(0);
    },
  );

  it.each([FakeWebSocket.CLOSING, FakeWebSocket.CLOSED])(
    "waits for the close event in state %s before applying transient backoff",
    async (state) => {
      const connected = api.connectWebSocket();
      const first = await socketAt();
      first.open();
      first.pong();
      await connected;
      first.beginClose();
      first.readyState = state;
      const outcome = api.subscribeToGraphExecution(executionID);
      await vi.advanceTimersByTimeAsync(0);
      expect(FakeWebSocket.instances).toHaveLength(1);
      first.serverClose(1013);
      expect(vi.getTimerCount()).toBe(1);
      await vi.advanceTimersByTimeAsync(999);
      expect(FakeWebSocket.instances).toHaveLength(1);
      await vi.advanceTimersByTimeAsync(1);
      const next = await socketAt(1);
      next.open();
      next.pong();
      await outcome;
      expect(next.send).toHaveBeenLastCalledWith(
        JSON.stringify({
          method: "subscribe_graph_execution",
          data: { graph_exec_id: executionID },
        }),
      );
      api.disconnectWebSocket();
      expect(vi.getTimerCount()).toBe(0);
    },
  );
});
