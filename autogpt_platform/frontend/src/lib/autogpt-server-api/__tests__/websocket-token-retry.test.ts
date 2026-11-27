import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/actions", () => ({ getWebSocketToken: vi.fn() }));
vi.mock("@/lib/auth/deployment-reload", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/deployment-reload")>()),
  promptForNewDeployment: vi.fn(),
  reloadForNewDeployment: vi.fn(() => true),
}));

import { getWebSocketToken } from "@/lib/auth/actions";
import {
  promptForNewDeployment,
  reloadForNewDeployment,
} from "@/lib/auth/deployment-reload";
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
    // The top of each jittered retry window: the full backoff step.
    vi.spyOn(Math, "random").mockReturnValue(1 - Number.EPSILON);
    vi.stubGlobal("WebSocket", FakeWebSocket);
    FakeWebSocket.instances = [];
    vi.mocked(getWebSocketToken)
      .mockReset()
      .mockResolvedValue({ token: "token" });
    vi.mocked(promptForNewDeployment).mockClear();
    vi.mocked(reloadForNewDeployment).mockClear();
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
      // Caught up (its ordering ping answered after the refetch spread) and
      // up long enough to count as healthy, so the backoff starts over.
      await vi.advanceTimersByTimeAsync(2000);
      next.pong();
      await vi.advanceTimersByTimeAsync(10_000);
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

  it.each([
    { visibility: "hidden", handOff: reloadForNewDeployment },
    { visibility: "visible", handOff: promptForNewDeployment },
  ] as const)(
    "hands a $visibility tab left on a replaced deployment to the new build instead of retrying",
    async ({ visibility, handOff }) => {
      vi.spyOn(document, "visibilityState", "get").mockReturnValue(visibility);
      const warn = vi.spyOn(console, "warn");
      vi.mocked(getWebSocketToken).mockRejectedValue(
        Object.assign(new Error("Server Action was not found"), {
          name: "UnrecognizedActionError",
        }),
      );
      const outcome = api.subscribeToGraphExecution(executionID).then(
        () => "sent",
        () => "rejected",
      );
      await vi.advanceTimersByTimeAsync(120_000);
      expect(getWebSocketToken).toHaveBeenCalledTimes(1);
      expect(vi.getTimerCount()).toBe(0);
      expect(handOff).toHaveBeenCalledTimes(1);
      const other =
        handOff === reloadForNewDeployment
          ? promptForNewDeployment
          : reloadForNewDeployment;
      expect(other).not.toHaveBeenCalled();
      expect(warn).not.toHaveBeenCalled();
      expect(await outcome).toBe("rejected");
      expect(FakeWebSocket.instances).toHaveLength(0);
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

  it.each([
    new Error("Network error"),
    Object.assign(new Error("Server Action was not found"), {
      name: "UnrecognizedActionError",
    }),
  ])(
    "does not retry or hand off a token failure arriving after logout: %s",
    async (failure) => {
      let rejectToken: ((error: Error) => void) | undefined;
      vi.mocked(getWebSocketToken).mockImplementationOnce(
        () =>
          new Promise((_, reject) => {
            rejectToken = reject;
          }),
      );
      const pending = api.connectWebSocket();
      setWebSocketDisconnectIntent();
      rejectToken?.(failure);
      await vi.advanceTimersByTimeAsync(60_000);
      await pending;
      expect(getWebSocketToken).toHaveBeenCalledTimes(1);
      expect(FakeWebSocket.instances).toHaveLength(0);
      expect(vi.getTimerCount()).toBe(0);
      expect(promptForNewDeployment).not.toHaveBeenCalled();
      expect(reloadForNewDeployment).not.toHaveBeenCalled();
    },
  );
});
