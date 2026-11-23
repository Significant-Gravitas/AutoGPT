import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/actions", () => ({ getWebSocketToken: vi.fn() }));

import { getWebSocketToken } from "@/lib/auth/actions";
import { subscribeTimedOutFrame } from "@/tests/integrations/subscribe-failure-frame";
import BackendAPI from "../client";
import type { GraphID } from "../types";
import { SubscribeRetries } from "../websocket-retry";
import { FakeWebSocket } from "./websocket-fake";

interface Request {
  method: string;
  data?: unknown;
}

const GRAPH_1: Request = {
  method: "subscribe_graph_executions",
  data: { graph_id: "graph-1" },
};

// What ws_api.py answers when Redis doesn't confirm a subscribe in time.
function couldNotComplete(request: Request) {
  return {
    method: "error",
    success: false,
    error: `Could not handle '${request.method}'; retry`,
    request,
    retryable: true,
  };
}

function refused(request: Request) {
  return {
    method: "error",
    success: false,
    error: "Subscription limit reached (1000 per connection)",
    request,
    retryable: false,
  };
}

// What a server from before the request and retryable fields answers.
function unattributed(request: Request) {
  return {
    method: "error",
    success: false,
    error: `Could not handle '${request.method}'; retry`,
  };
}

function confirmed(request: Request) {
  return { method: request.method, success: true, channel: "user-1|graph" };
}

function failing(
  times: number,
  failure: (request: Request) => unknown = couldNotComplete,
) {
  let failures = 0;
  return (request: Request) =>
    failures++ < times ? failure(request) : confirmed(request);
}

function failingFor(graphID: string, times: number) {
  let failures = 0;
  return (request: Request) =>
    (request.data as { graph_id?: string }).graph_id === graphID &&
    failures++ < times
      ? couldNotComplete(request)
      : confirmed(request);
}

// Answers each message as the server does: in order, so the pong to a ping
// comes after the answer to every subscribe sent before it. A string answer
// is sent exactly as it is.
function serve(socket: FakeWebSocket, answer: (request: Request) => unknown) {
  socket.send.mockImplementation((data: string) => {
    const request: Request = JSON.parse(data);
    // Signing in gets no answer of its own: the pong after it confirms it.
    if (request.method === "authenticate") return;
    const reply =
      request.method === "heartbeat"
        ? { method: "heartbeat", data: "pong" }
        : answer(request);
    queueMicrotask(() =>
      typeof reply === "string"
        ? socket.onmessage?.(new MessageEvent("message", { data: reply }))
        : socket.receive(reply),
    );
  });
}

// What the socket sent after signing in.
function sent(socket: FakeWebSocket): Request[] {
  return socket.send.mock.calls
    .map(([data]) => JSON.parse(data))
    .filter((message) => message.method !== "authenticate");
}

function subscribes(socket: FakeWebSocket) {
  return sent(socket).filter((message) => message.method !== "heartbeat");
}

describe("a subscribe the server could not complete", () => {
  let api: BackendAPI;
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", FakeWebSocket);
    FakeWebSocket.instances = [];
    vi.mocked(getWebSocketToken)
      .mockReset()
      .mockResolvedValue({ token: "test-token" });
    // The low end of each jittered wait, half its step; no reconnect spread.
    vi.spyOn(Math, "random").mockReturnValue(0);
    api = new BackendAPI("http://test", "ws://test/ws");
  });
  afterEach(() => {
    api.disconnectWebSocket();
    vi.clearAllTimers();
    vi.unstubAllGlobals();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  async function socketAt(index: number) {
    await vi.waitFor(
      () => expect(FakeWebSocket.instances.length).toBeGreaterThan(index),
      { interval: 1 },
    );
    return FakeWebSocket.instances[index];
  }

  function viewSubscribes() {
    api.onWebSocketConnect(
      () => void api.subscribeToGraphExecutions("graph-1" as GraphID),
    );
  }

  async function accept(index: number, answer: (request: Request) => unknown) {
    const socket = await socketAt(index);
    serve(socket, answer);
    socket.open();
    await vi.advanceTimersByTimeAsync(0);
    return socket;
  }

  async function reconnect(answer: (request: Request) => unknown) {
    const reconnected = vi.fn();
    api.onWebSocketReconnect(reconnected);
    viewSubscribes();
    const first = await accept(0, failing(0));
    first.serverClose(1011);
    await vi.advanceTimersByTimeAsync(500);
    const second = await accept(1, answer);
    return { second, reconnected };
  }

  it("is sent again, twice as late after each failure, up to five times", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    viewSubscribes();
    const socket = await accept(0, failing(Infinity));
    expect(subscribes(socket)).toEqual([GRAPH_1]);

    for (const [retry, delay] of [500, 1000, 2000, 4000, 8000].entries()) {
      await vi.advanceTimersByTimeAsync(delay - 1);
      expect(subscribes(socket)).toHaveLength(retry + 1);
      await vi.advanceTimersByTimeAsync(1);
      expect(subscribes(socket)).toEqual(Array(retry + 2).fill(GRAPH_1));
    }
    await vi.advanceTimersByTimeAsync(60_000);
    expect(subscribes(socket)).toHaveLength(6);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toContain("subscribe_graph_executions");
  });

  it("warns once per connection when retries run out, however many do", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const info = vi.spyOn(console, "info").mockImplementation(() => {});
    viewSubscribes();
    api.onWebSocketConnect(
      () => void api.subscribeToGraphExecutions("graph-2" as GraphID),
    );
    await accept(0, failing(Infinity));

    await vi.advanceTimersByTimeAsync(60_000);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(info).toHaveBeenCalledWith(
      expect.stringContaining("still failing after 5 retries"),
    );
  });

  it("warns again on the next connection when retries run out there too", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    function givingUp() {
      return warn.mock.calls.filter(([message]) =>
        String(message).includes("still failing after"),
      ).length;
    }
    viewSubscribes();
    const first = await accept(0, failing(Infinity));
    await vi.advanceTimersByTimeAsync(60_000);
    expect(givingUp()).toBe(1);

    first.serverClose(1011);
    await vi.advanceTimersByTimeAsync(500);
    await accept(1, failing(Infinity));
    await vi.advanceTimersByTimeAsync(60_000);
    expect(givingUp()).toBe(2);
  });

  it("gets all its retries again if it fails after a retry was confirmed", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const answers: ((request: Request) => unknown)[] = [
      ...Array.from({ length: 5 }, () => couldNotComplete),
      confirmed,
      couldNotComplete,
    ];
    viewSubscribes();
    const socket = await accept(0, (request) =>
      (answers.shift() ?? confirmed)(request),
    );
    // Confirmed on its fifth and last retry.
    await vi.advanceTimersByTimeAsync(15_500);
    expect(subscribes(socket)).toHaveLength(6);

    void api.subscribeToGraphExecutions("graph-1" as GraphID);
    await vi.advanceTimersByTimeAsync(500);
    expect(subscribes(socket)).toHaveLength(8);
    expect(warn).not.toHaveBeenCalled();
  });

  it("is sent again after the exact frame the realtime journey injects", async () => {
    viewSubscribes();
    let answers = 0;
    const socket = await accept(0, (request) =>
      answers++ === 0 ? subscribeTimedOutFrame("graph-1") : confirmed(request),
    );
    expect(subscribes(socket)).toEqual([GRAPH_1]);

    await vi.advanceTimersByTimeAsync(500);
    expect(subscribes(socket)).toEqual([GRAPH_1, GRAPH_1]);
  });

  it("is not sent again once the server refuses it, and says why", async () => {
    const info = vi.spyOn(console, "info").mockImplementation(() => {});
    viewSubscribes();
    const socket = await accept(0, failing(Infinity, refused));

    await vi.advanceTimersByTimeAsync(60_000);
    expect(subscribes(socket)).toEqual([GRAPH_1]);
    expect(info).toHaveBeenCalledWith(
      expect.stringContaining("Subscription limit reached"),
    );
  });

  it("is not sent again when another view's copy of it is refused", async () => {
    vi.spyOn(console, "info").mockImplementation(() => {});
    viewSubscribes();
    viewSubscribes();
    let answers = 0;
    const socket = await accept(0, (request) =>
      answers++ === 0 ? couldNotComplete(request) : refused(request),
    );
    expect(subscribes(socket)).toHaveLength(2);

    await vi.advanceTimersByTimeAsync(60_000);
    expect(subscribes(socket)).toHaveLength(2);
  });

  it("is sent again once, however many views asked for it", async () => {
    viewSubscribes();
    viewSubscribes();
    const socket = await accept(0, failing(2));
    expect(subscribes(socket)).toHaveLength(2);

    await vi.advanceTimersByTimeAsync(60_000);
    expect(subscribes(socket)).toHaveLength(3);
  });

  it("lets the views re-read at once after a reconnect, and again once its retry is confirmed", async () => {
    // A view whose subscription goes through, beside the one that fails.
    api.onWebSocketConnect(
      () => void api.subscribeToGraphExecutions("graph-2" as GraphID),
    );
    const { second, reconnected } = await reconnect(failingFor("graph-1", 1));
    // The pong to the ordering ping is in: graph-2's views are fresh, while
    // graph-1 waits for its retry.
    expect(sent(second).map((message) => message.method)).toEqual([
      "heartbeat",
      "subscribe_graph_executions",
      "subscribe_graph_executions",
      "heartbeat",
    ]);
    expect(reconnected).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(499);
    expect(reconnected).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    // The retry, then a ping whose pong follows the retry's answer.
    expect(
      sent(second)
        .map((message) => message.method)
        .slice(4),
    ).toEqual(["subscribe_graph_executions", "heartbeat"]);
    expect(reconnected).toHaveBeenCalledTimes(2);
  });

  it("sent during the reconnect spread, leaves the ordering ping to after it", async () => {
    // A 1 s spread; the first retry waits 750 ms.
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const { second, reconnected } = await reconnect(failing(1));

    await vi.advanceTimersByTimeAsync(1000);
    expect(sent(second).map((message) => message.method)).toEqual([
      "heartbeat",
      "subscribe_graph_executions",
      "subscribe_graph_executions",
      "heartbeat",
    ]);
    expect(reconnected).toHaveBeenCalledTimes(1);
  });

  it("lets the views re-read again after a reconnect once its retry is refused", async () => {
    let answers = 0;
    const { reconnected } = await reconnect((request) =>
      answers++ === 0 ? couldNotComplete(request) : refused(request),
    );
    vi.spyOn(console, "info").mockImplementation(() => {});
    expect(reconnected).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(500);
    expect(reconnected).toHaveBeenCalledTimes(2);
  });

  it("from a server that doesn't name the request changes nothing", async () => {
    const { second, reconnected } = await reconnect(
      failing(Infinity, unattributed),
    );
    expect(reconnected).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(60_000);
    expect(subscribes(second)).toEqual([GRAPH_1]);
  });

  it("on the first connection, lets the views re-read once its retry is confirmed", async () => {
    const reconnected = vi.fn();
    api.onWebSocketReconnect(reconnected);
    viewSubscribes();
    const socket = await accept(0, failing(1));
    expect(reconnected).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(500);
    expect(subscribes(socket)).toEqual([GRAPH_1, GRAPH_1]);
    expect(reconnected).toHaveBeenCalledTimes(1);
  });

  it("is dropped with its connection", async () => {
    viewSubscribes();
    // Two retries fail, so the next waits 2 s: past the reconnect below.
    const first = await accept(0, failing(Infinity));
    await vi.advanceTimersByTimeAsync(1500);
    expect(subscribes(first)).toHaveLength(3);
    first.serverClose(1011);
    await vi.advanceTimersByTimeAsync(500);
    const second = await accept(1, failing(0));

    await vi.advanceTimersByTimeAsync(10_000);
    // The view subscribes again on the new connection, and only the view.
    expect(subscribes(second)).toEqual([GRAPH_1]);
    api.disconnectWebSocket();
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe("SubscribeRetries", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("takes echoes of one subscribe with their keys in another order as one", async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0);
    const resend = vi.fn();
    const retries = new SubscribeRetries(resend);
    for (const data of [
      { graph_id: "graph-1", after: "x" },
      { after: "x", graph_id: "graph-1" },
    ]) {
      retries.read(
        couldNotComplete({ method: "subscribe_graph_executions", data }),
      );
    }

    await vi.advanceTimersByTimeAsync(60_000);
    expect(resend).toHaveBeenCalledTimes(1);
    retries.clear();
  });
});
