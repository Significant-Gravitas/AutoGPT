import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/actions", () => ({
  getWebSocketToken: vi.fn(async () => ({ token: "test-token" })),
}));

import {
  clearWebSocketDisconnectIntent,
  setWebSocketDisconnectIntent,
} from "@/lib/auth/helpers";
import BackendAPI from "../client";

// Logging out closes the socket on purpose: in the tab that logs out, and
// through the shared disconnect intent in every other open tab. Sentry turns
// each console.error and console.warn into an event, so such a close has to
// stay below warn, while a close nobody asked for keeps being reported.
describe("BackendAPI WebSocket closed by a logout", () => {
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
    onerror: (() => void) | null = null;
    onmessage: ((event: MessageEvent) => void) | null = null;
    state = "connecting";
    close = vi.fn();
    send = vi.fn();

    constructor(public url: string) {
      sockets.push(this);
    }
  }

  afterEach(() => {
    sockets.length = 0;
    clearWebSocketDisconnectIntent();
    vi.unstubAllGlobals();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  async function connect() {
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", FakeWebSocket);
    const api = new BackendAPI("http://test", "ws://test/ws");
    void api.connectWebSocket();
    await vi.waitFor(() => expect(sockets).toHaveLength(1), { interval: 1 });
    return { api, socket: sockets[0] };
  }

  function spyOnConsole() {
    return {
      error: vi.spyOn(console, "error").mockImplementation(() => undefined),
      warn: vi.spyOn(console, "warn").mockImplementation(() => undefined),
      debug: vi.spyOn(console, "debug").mockImplementation(() => undefined),
    };
  }

  it("keeps a socket this tab's logout closed mid-connect out of the error log", async () => {
    const logs = spyOnConsole();
    const { api, socket } = await connect();

    api.disconnectWebSocket();
    socket.onclose!({ code: 1006, reason: "", wasClean: false });

    expect(socket.close).toHaveBeenCalledTimes(1);
    expect(logs.error).not.toHaveBeenCalled();
    expect(logs.warn).not.toHaveBeenCalled();
    expect(logs.debug).toHaveBeenCalledWith(
      expect.stringContaining("code 1006 (abnormal closure)"),
    );
    await vi.runAllTimersAsync();
    expect(sockets).toHaveLength(1);
  });

  it("keeps an open socket another tab's logout closed out of the warn log", async () => {
    const logs = spyOnConsole();
    const { socket } = await connect();
    socket.state = "connected";

    setWebSocketDisconnectIntent();
    socket.onclose!({ code: 1005, reason: "", wasClean: true });

    expect(logs.error).not.toHaveBeenCalled();
    expect(logs.warn).not.toHaveBeenCalled();
    await vi.runAllTimersAsync();
    expect(sockets).toHaveLength(1);
  });

  it("still reports a socket that fails to connect with nobody logging out, and reconnects", async () => {
    const logs = spyOnConsole();
    const { socket } = await connect();

    socket.onclose!({ code: 1006, reason: "", wasClean: false });

    expect(logs.error).toHaveBeenCalledTimes(1);
    expect(logs.error.mock.calls[0][0]).toContain(
      "[BackendAPI] WebSocket failed to connect: code 1006",
    );
    await vi.runAllTimersAsync();
    await vi.waitFor(() => expect(sockets).toHaveLength(2), { interval: 1 });
  });
});
