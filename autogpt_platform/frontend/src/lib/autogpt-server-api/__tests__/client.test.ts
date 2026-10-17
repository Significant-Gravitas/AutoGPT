import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/actions", () => ({
  getWebSocketToken: vi.fn(async () => ({ token: "test-token" })),
}));

import BackendAPI, {
  buildOAuthLoginQuery,
  getWSReconnectDelay,
  WebSocketSubscriptionInterrupted,
} from "../client";
import type { GraphID } from "../types";

describe("BackendAPI.oAuthLogin", () => {
  it("passes credentialID through to buildOAuthLoginQuery", async () => {
    const api = new BackendAPI("http://test", "ws://test");
    const spy = vi.spyOn(api as any, "_get").mockResolvedValue({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-abc",
    });

    const result = await api.oAuthLogin("google", ["drive.file"], "cred-1");

    expect(spy).toHaveBeenCalledWith("/integrations/google/login", {
      scopes: "drive.file",
      credential_id: "cred-1",
    });
    expect(result).toEqual({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-abc",
    });
  });

  it("omits query when no scopes or credentialID", async () => {
    const api = new BackendAPI("http://test", "ws://test");
    const spy = vi
      .spyOn(api as any, "_get")
      .mockResolvedValue({ login_url: "url", state_token: "tok" });

    await api.oAuthLogin("github");

    expect(spy).toHaveBeenCalledWith("/integrations/github/login", undefined);
  });
});

describe("buildOAuthLoginQuery", () => {
  it("returns undefined when called with no args", () => {
    expect(buildOAuthLoginQuery()).toBeUndefined();
  });

  it("returns undefined when scopes is empty and credentialID is absent", () => {
    // Old behavior sent `{scopes: ""}` for an empty array, which the
    // backend rejects. Pin the tighter contract.
    expect(buildOAuthLoginQuery([])).toBeUndefined();
  });

  it("joins scopes with a comma", () => {
    expect(buildOAuthLoginQuery(["drive.file", "drive.metadata"])).toEqual({
      scopes: "drive.file,drive.metadata",
    });
  });

  it("includes credential_id when provided", () => {
    expect(buildOAuthLoginQuery(undefined, "cred-1")).toEqual({
      credential_id: "cred-1",
    });
  });

  it("includes both scopes and credential_id when both are provided", () => {
    // The incremental-OAuth flow sends both: the scopes the block needs,
    // plus the credential to merge them into.
    expect(buildOAuthLoginQuery(["drive.file"], "cred-1")).toEqual({
      scopes: "drive.file",
      credential_id: "cred-1",
    });
  });

  it("ignores an empty credentialID", () => {
    expect(buildOAuthLoginQuery(["drive.file"], "")).toEqual({
      scopes: "drive.file",
    });
  });
});

describe("BackendAPI._makeClientRequest 204 handling", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  function mockFetch(response: Response) {
    return vi.spyOn(globalThis, "fetch").mockResolvedValue(response);
  }

  it("returns null for 204 No Content without parsing body", async () => {
    const api = new BackendAPI("http://test", "ws://test");
    mockFetch(new Response(null, { status: 204 }));

    const result = await (api as any)._makeClientRequest(
      "DELETE",
      "/library/agents/abc",
    );

    expect(result).toBeNull();
  });

  it("returns null when Content-Length is 0", async () => {
    const api = new BackendAPI("http://test", "ws://test");
    mockFetch(
      new Response("", { status: 200, headers: { "Content-Length": "0" } }),
    );

    const result = await (api as any)._makeClientRequest("DELETE", "/x");

    expect(result).toBeNull();
  });

  it("parses JSON body for non-empty 200 responses", async () => {
    const api = new BackendAPI("http://test", "ws://test");
    mockFetch(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const result = await (api as any)._makeClientRequest("GET", "/x");

    expect(result).toEqual({ ok: true });
  });
});

describe("BackendAPI WebSocket failure logging", () => {
  const sockets: FakeWebSocket[] = [];

  class FakeWebSocket {
    static readonly CONNECTING = 0;
    static readonly OPEN = 1;
    static readonly CLOSING = 2;
    static readonly CLOSED = 3;

    readyState = FakeWebSocket.CONNECTING;
    onopen: (() => void) | null = null;
    onclose:
      | ((event: Pick<CloseEvent, "code" | "reason" | "wasClean">) => void)
      | null = null;
    onerror:
      | ((event: Pick<Event, "type"> & { target?: unknown }) => void)
      | null = null;
    onmessage: ((event: MessageEvent) => void) | null = null;
    state = "connecting";
    close = vi.fn();
    send = vi.fn();

    constructor(public url: string) {
      sockets.push(this);
    }
  }

  const apis: BackendAPI[] = [];

  afterEach(() => {
    // Stop earlier instances reacting to the document events later tests fire.
    apis.splice(0).forEach((api) => api.disconnectWebSocket());
    sockets.length = 0;
    vi.unstubAllGlobals();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  const tokenFetch = vi.fn(
    async () => new Response(JSON.stringify({ token: "test-token" })),
  );

  async function connect() {
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", FakeWebSocket);
    tokenFetch.mockClear();
    vi.stubGlobal("fetch", tokenFetch);
    const api = new BackendAPI("http://test", "ws://test/ws");
    apis.push(api);
    void api.connectWebSocket();
    await vi.waitFor(() => expect(sockets).toHaveLength(1), { interval: 1 });
    return sockets[0];
  }

  it("reports the close code and reason instead of the CloseEvent object", async () => {
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    const socket = await connect();

    socket.onclose!({ code: 4002, reason: "Invalid token", wasClean: true });

    expect(consoleError).toHaveBeenCalledTimes(1);
    const [message, ...rest] = consoleError.mock.calls[0];
    expect(rest).toHaveLength(0);
    expect(message).toContain("[BackendAPI] WebSocket failed to connect");
    expect(message).toContain("4002");
    expect(message).toContain("Invalid token");
    expect(message).not.toContain("[object");
  });

  it("reports a close on an established connection at warn level", async () => {
    const consoleWarn = vi
      .spyOn(console, "warn")
      .mockImplementation(() => undefined);
    const socket = await connect();
    socket.state = "connected";

    socket.onclose!({ code: 1006, reason: "", wasClean: false });

    const [message] = consoleWarn.mock.calls[0];
    expect(message).toContain("[BackendAPI] WebSocket connection closed");
    expect(message).toContain("1006");
    expect(message).toContain("abnormal closure");
    expect(message).not.toContain("[object");
  });

  it("reports the socket state on an error event instead of the Event object", async () => {
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    const socket = await connect();
    socket.state = "connected";
    socket.readyState = FakeWebSocket.CLOSED;

    socket.onerror!({ type: "error", target: socket });

    const [message, ...rest] = consoleError.mock.calls[0];
    expect(rest).toHaveLength(0);
    expect(message).toContain("[BackendAPI] WebSocket error");
    expect(message).toContain("CLOSED");
    expect(message).not.toContain("[object");
    expect(message).not.toContain("test-token");
  });

  it("keeps retrying the reconnect when fetching a token fails", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    vi.spyOn(console, "info").mockImplementation(() => undefined);
    const socket = await connect();
    socket.state = "connected";
    socket.onopen!();

    tokenFetch
      .mockClear()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(new Response("", { status: 502 }));
    socket.onclose!({ code: 1006, reason: "", wasClean: false });

    // A network failure and a 5xx, each followed by a backed-off retry,
    // then a fresh socket — never a dead client waiting for a page reload.
    await vi.waitFor(
      async () => {
        await vi.advanceTimersByTimeAsync(30_000);
        expect(sockets).toHaveLength(2);
      },
      { interval: 1 },
    );
    expect(tokenFetch).toHaveBeenCalledTimes(3);
    expect(tokenFetch).toHaveBeenCalledWith(
      "/api/auth/token",
      expect.anything(),
    );
  });

  it("does not retry when there is no session", async () => {
    vi.spyOn(console, "info").mockImplementation(() => undefined);
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", FakeWebSocket);
    tokenFetch.mockClear();
    tokenFetch.mockResolvedValue(new Response("", { status: 401 }));
    vi.stubGlobal("fetch", tokenFetch);
    const api = new BackendAPI("http://test", "ws://test/ws");
    apis.push(api);

    await api.connectWebSocket();
    await vi.advanceTimersByTimeAsync(60_000);

    expect(sockets).toHaveLength(0);
    expect(tokenFetch).toHaveBeenCalledTimes(1);
    tokenFetch.mockReset();
    tokenFetch.mockImplementation(
      async () => new Response(JSON.stringify({ token: "test-token" })),
    );
  });

  it("reconnects at once when the browser comes back online", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const socket = await connect();
    socket.state = "connected";
    socket.onopen!();
    socket.onclose!({ code: 1006, reason: "", wasClean: false });

    // A reconnect is now waiting out its backoff; going online skips it.
    window.dispatchEvent(new Event("online"));

    await vi.waitFor(() => expect(sockets).toHaveLength(2), { interval: 1 });
  });

  it("probes an open socket when the tab becomes visible again", async () => {
    const socket = await connect();
    socket.readyState = FakeWebSocket.OPEN;
    socket.onopen!();

    document.dispatchEvent(new Event("visibilitychange"));

    expect(socket.send).toHaveBeenCalledWith(
      expect.stringContaining('"heartbeat"'),
    );
  });

  it("resolves a subscribe only once the server confirms it", async () => {
    const socket = await connect();
    socket.readyState = FakeWebSocket.OPEN;
    socket.onopen!();
    const api = apis.at(-1)!;

    let confirmed = false;
    const subscribed = api
      .subscribeToGraphExecutions("graph-1" as GraphID)
      .then(() => (confirmed = true));
    await vi.waitFor(() => expect(socket.send).toHaveBeenCalled(), {
      interval: 1,
    });
    await Promise.resolve();
    expect(confirmed).toBe(false);

    socket.onmessage!({
      data: JSON.stringify({
        method: "subscribe_graph_executions",
        success: true,
        channel: "user-1|graph#graph-1|executions",
      }),
    } as MessageEvent);
    await subscribed;
    expect(confirmed).toBe(true);
  });

  it("fails a pending subscribe as interrupted when the socket closes", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const socket = await connect();
    socket.readyState = FakeWebSocket.OPEN;
    socket.state = "connected";
    socket.onopen!();
    const subscribed = apis
      .at(-1)!
      .subscribeToGraphExecutions("graph-1" as GraphID);
    await vi.waitFor(() => expect(socket.send).toHaveBeenCalled(), {
      interval: 1,
    });

    socket.onclose!({ code: 1011, reason: "", wasClean: true });

    await expect(subscribed).rejects.toBeInstanceOf(
      WebSocketSubscriptionInterrupted,
    );
  });

  it("one ack confirms only the oldest pending subscribe for its channel", async () => {
    const socket = await connect();
    socket.readyState = FakeWebSocket.OPEN;
    socket.onopen!();
    const api = apis.at(-1)!;
    const ack = () =>
      socket.onmessage!({
        data: JSON.stringify({
          method: "subscribe_graph_executions",
          success: true,
          channel: "user-1|graph#graph-1|executions",
        }),
      } as MessageEvent);

    let secondConfirmed = false;
    const first = api.subscribeToGraphExecutions("graph-1" as GraphID);
    const second = api
      .subscribeToGraphExecutions("graph-1" as GraphID)
      .then(() => (secondConfirmed = true));
    await vi.waitFor(() => expect(socket.send).toHaveBeenCalledTimes(2), {
      interval: 1,
    });

    ack();
    await first;
    await vi.advanceTimersByTimeAsync(0);
    expect(secondConfirmed).toBe(false);

    ack();
    await second;
    expect(secondConfirmed).toBe(true);
  });

  it("a failed send fails only its own subscribe", async () => {
    const socket = await connect();
    socket.readyState = FakeWebSocket.OPEN;
    socket.onopen!();
    const api = apis.at(-1)!;
    const unhandled = vi.fn();
    process.on("unhandledRejection", unhandled);

    const first = api.subscribeToGraphExecutions("graph-1" as GraphID);
    await vi.waitFor(() => expect(socket.send).toHaveBeenCalledTimes(1), {
      interval: 1,
    });
    socket.send.mockImplementationOnce(() => {
      throw new Error("send failed");
    });
    await expect(
      api.subscribeToGraphExecutions("graph-2" as GraphID),
    ).rejects.toThrow("send failed");

    // The first subscription's confirmation still lands.
    socket.onmessage!({
      data: JSON.stringify({
        method: "subscribe_graph_executions",
        success: true,
        channel: "user-1|graph#graph-1|executions",
      }),
    } as MessageEvent);
    await expect(first).resolves.toBeUndefined();
    await vi.advanceTimersByTimeAsync(0);
    process.off("unhandledRejection", unhandled);
    expect(unhandled).not.toHaveBeenCalled();
  });

  it("keeps backing off when sockets open and drop straight away", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    vi.spyOn(Math, "random").mockReturnValue(1);
    const first = await connect();
    first.state = "connected";
    first.onopen!();
    first.onclose!({ code: 1011, reason: "", wasClean: true });
    await vi.advanceTimersByTimeAsync(1_000);
    await vi.waitFor(() => expect(sockets).toHaveLength(2), { interval: 1 });

    // Opens, but drops before it has been healthy for long.
    const second = sockets[1];
    second.state = "connected";
    second.onopen!();
    second.onclose!({ code: 1011, reason: "", wasClean: true });

    await vi.advanceTimersByTimeAsync(1_500);
    expect(sockets).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(600);
    await vi.waitFor(() => expect(sockets).toHaveLength(3), { interval: 1 });
  });

  it("backs off reconnects exponentially, with jitter, up to a cap", () => {
    vi.spyOn(Math, "random").mockReturnValue(1);
    expect(getWSReconnectDelay(0)).toBe(1_000);
    expect(getWSReconnectDelay(3)).toBe(8_000);
    expect(getWSReconnectDelay(10)).toBe(30_000);
    vi.spyOn(Math, "random").mockReturnValue(0);
    expect(getWSReconnectDelay(10)).toBe(15_000);
  });
});
