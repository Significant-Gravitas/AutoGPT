import { FakeWebSocket } from "@/lib/autogpt-server-api/__tests__/websocket-fake";
import { act, waitFor } from "@testing-library/react";
import { expect, vi } from "vitest";

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

  return {
    /** Accepts the app's next connection and keeps answering its pings. */
    async acceptNext() {
      await waitFor(
        () => expect(FakeWebSocket.instances.length).toBeGreaterThan(accepted),
        { timeout: 5000 },
      );
      const socket = FakeWebSocket.instances[accepted];
      accepted += 1;
      socket.send.mockImplementation((data: string) => {
        if (JSON.parse(data).method === "heartbeat") {
          queueMicrotask(() => socket.pong());
        }
      });
      await act(async () => socket.open());
      return socket;
    },
    /** What the server does when it loses its event stream. */
    async drop(socket: FakeWebSocket) {
      await act(async () => socket.serverClose(1011));
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
