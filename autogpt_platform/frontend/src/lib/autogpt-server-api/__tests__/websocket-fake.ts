import { vi } from "vitest";

export class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: FakeWebSocket[] = [];
  readyState = FakeWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onclose:
    | ((event: Pick<CloseEvent, "code" | "reason" | "wasClean">) => void)
    | null = null;
  onerror:
    | ((event: Pick<Event, "type"> & { target?: unknown }) => void)
    | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  close = vi.fn(() => this.beginClose());
  send = vi.fn();

  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }
  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }
  pong() {
    this.onmessage?.(
      new MessageEvent("message", {
        data: JSON.stringify({ method: "heartbeat", data: "pong" }),
      }),
    );
  }
  beginClose() {
    this.readyState = FakeWebSocket.CLOSING;
  }
  serverClose(code: number) {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code, reason: "", wasClean: code !== 1006 });
  }
}
