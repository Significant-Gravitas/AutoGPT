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

describe("WebSocket token issuance recovery", () => {
  let api: BackendAPI;
  const executionID =
    "00000000-0000-4000-8000-000000000001" as GraphExecutionID;
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", FakeWebSocket);
    FakeWebSocket.instances = [];
    vi.mocked(getWebSocketToken)
      .mockReset()
      .mockResolvedValue({ token: "token" });
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

  it.each(["returned", "thrown"])(
    "recovers existing subscribers and pending sends after a %s token failure",
    async (failure) => {
      const connected = vi.fn();
      api.onWebSocketConnect(connected);
      await vi.advanceTimersByTimeAsync(0);
      const first = FakeWebSocket.instances[0];
      first.open();
      first.pong();
      if (failure === "returned") {
        vi.mocked(getWebSocketToken).mockResolvedValueOnce({
          token: null,
          error: "Authentication service unavailable",
          retryable: true,
        });
      } else {
        vi.mocked(getWebSocketToken).mockRejectedValueOnce(
          new TypeError("Network error"),
        );
      }
      first.serverClose(1013);
      const sent = vi.fn();
      const rejected = vi.fn();
      const pending = api
        .subscribeToGraphExecution(executionID)
        .then(sent, rejected);
      await vi.advanceTimersByTimeAsync(1000);
      expect(getWebSocketToken).toHaveBeenCalledTimes(2);
      expect(sent).not.toHaveBeenCalled();
      expect(rejected).not.toHaveBeenCalled();
      await vi.advanceTimersByTimeAsync(1999);
      expect(FakeWebSocket.instances).toHaveLength(1);
      await vi.advanceTimersByTimeAsync(1);
      expect(FakeWebSocket.instances).toHaveLength(2);
      const next = FakeWebSocket.instances[1];
      next.open();
      expect(sent).not.toHaveBeenCalled();
      next.pong();
      await pending;
      expect(sent).toHaveBeenCalledTimes(1);
      expect(connected).toHaveBeenCalledTimes(2);
      expect(next.send).toHaveBeenLastCalledWith(
        JSON.stringify({
          method: "subscribe_graph_execution",
          data: { graph_exec_id: executionID },
        }),
      );
      next.serverClose(1006);
      await vi.advanceTimersByTimeAsync(1000);
      expect(FakeWebSocket.instances).toHaveLength(3);
    },
  );

  it.each([
    { token: null },
    { token: "rejected-token", error: "Session rejected", retryable: false },
  ])(
    "stops retrying when token issuance rejects the session: %j",
    async (result) => {
      vi.mocked(getWebSocketToken).mockResolvedValueOnce(result);
      const outcome = api.subscribeToGraphExecution(executionID).then(
        () => "sent",
        () => "rejected",
      );
      await vi.advanceTimersByTimeAsync(60_000);
      expect(await outcome).toBe("rejected");
      expect(getWebSocketToken).toHaveBeenCalledTimes(1);
      expect(FakeWebSocket.instances).toHaveLength(0);
      expect(vi.getTimerCount()).toBe(0);
    },
  );

  it("caps token-error retries at thirty seconds and cancels them on disconnect", async () => {
    vi.mocked(getWebSocketToken).mockRejectedValue(
      new TypeError("Network error"),
    );
    const pending = api.connectWebSocket();
    await vi.advanceTimersByTimeAsync(0);
    const delays = [1000, 2000, 4000, 8000, 16000, 30000, 30000];
    for (const [index, delay] of delays.entries()) {
      await vi.advanceTimersByTimeAsync(delay - 1);
      expect(getWebSocketToken).toHaveBeenCalledTimes(index + 1);
      await vi.advanceTimersByTimeAsync(1);
      expect(getWebSocketToken).toHaveBeenCalledTimes(index + 2);
    }
    api.disconnectWebSocket();
    await pending;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(getWebSocketToken).toHaveBeenCalledTimes(delays.length + 1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("does not retry a token failure arriving after logout", async () => {
    let rejectToken: ((error: Error) => void) | undefined;
    vi.mocked(getWebSocketToken).mockImplementationOnce(
      () =>
        new Promise((_, reject) => {
          rejectToken = reject;
        }),
    );
    const pending = api.connectWebSocket();
    setWebSocketDisconnectIntent();
    rejectToken?.(new Error("Network error"));
    await vi.advanceTimersByTimeAsync(60_000);
    await pending;
    expect(getWebSocketToken).toHaveBeenCalledTimes(1);
    expect(FakeWebSocket.instances).toHaveLength(0);
    expect(vi.getTimerCount()).toBe(0);
  });
});
