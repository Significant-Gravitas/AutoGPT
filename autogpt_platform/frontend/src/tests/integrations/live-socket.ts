import { FakeWebSocket } from "@/lib/autogpt-server-api/__tests__/websocket-fake";
import BackendAPI from "@/lib/autogpt-server-api/client";
import { act, waitFor } from "@testing-library/react";
import { expect, onTestFinished, vi } from "vitest";

/**
 * How long to wait for a view to change after a reconnect: the app spreads
 * its refetches over up to 2 s, and the request comes on top.
 */
export const AFTER_RECONNECT = { timeout: 6000 };

export interface SentMessage {
  method: string;
  data?: unknown;
}

/**
 * Stands in for the backend's run-updates WebSocket. The app's real
 * BackendAPI connects through it and the fake answers every heartbeat as
 * the server does, so a page test can drop the connection and let the app
 * reconnect on its own. The test file also mocks `getWebSocketToken` to
 * return a token, or the app never opens a socket.
 */
export function installLiveSocket() {
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket);
  let accepted = 0;
  const notices = countReconnectNotices();
  let noticesAtDrop = 0;

  return {
    /**
     * Accepts the app's next connection and keeps answering its pings.
     * `answer` gets every other message the app sends and replies with
     * `socket.receive`. Like the server, the socket answers in order: each
     * reply, pongs included, waits for those before it, so an answer that
     * returns a promise holds every answer after it.
     */
    async acceptNext(
      answer?: (
        message: SentMessage,
        socket: FakeWebSocket,
      ) => void | Promise<void>,
    ) {
      await waitFor(
        () => expect(FakeWebSocket.instances.length).toBeGreaterThan(accepted),
        { timeout: 5000 },
      );
      const socket = FakeWebSocket.instances[accepted];
      accepted += 1;
      let answered = Promise.resolve();
      socket.send.mockImplementation((data: string) => {
        const message: SentMessage = JSON.parse(data);
        answered = answered.then(() =>
          message.method === "heartbeat"
            ? socket.pong()
            : answer?.(message, socket),
        );
      });
      await act(async () => socket.open());
      return socket;
    },
    /** What the server does when it loses its event stream. */
    async drop(socket: FakeWebSocket) {
      noticesAtDrop = notices.count();
      await act(async () => socket.serverClose(1011));
    },
    /**
     * Waits until the app has told its views it reconnected after the last
     * drop, so every reconnect handler has run.
     */
    async caughtUp() {
      await waitFor(
        () => expect(notices.count()).toBeGreaterThan(noticesAtDrop),
        AFTER_RECONNECT,
      );
      await act(async () => {});
    },
    /** Delivers a run update over the socket, as the server publishes one. */
    async publish(socket: FakeWebSocket, execution: Record<string, unknown>) {
      await act(async () =>
        socket.onmessage?.(
          new MessageEvent("message", {
            data: JSON.stringify({
              method: "graph_execution_event",
              data: execution,
            }),
          }),
        ),
      );
    },
  };
}

// Each BackendAPI the app registers a reconnect handler on gets one more,
// first, which counts the reconnect notices.
function countReconnectNotices() {
  let notices = 0;
  const counted = new WeakSet<BackendAPI>();
  const register = BackendAPI.prototype.onWebSocketReconnect;
  const spy = vi
    .spyOn(BackendAPI.prototype, "onWebSocketReconnect")
    .mockImplementation(function (this: BackendAPI, handler: () => void) {
      if (!counted.has(this)) {
        counted.add(this);
        register.call(this, () => {
          notices += 1;
        });
      }
      return register.call(this, handler);
    });
  onTestFinished(() => spy.mockRestore());
  return { count: () => notices };
}
