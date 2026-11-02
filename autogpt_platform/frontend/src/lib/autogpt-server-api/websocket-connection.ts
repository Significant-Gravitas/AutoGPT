import { getWebSocketToken } from "@/lib/auth/actions";
import { hasWebSocketDisconnectIntent } from "@/lib/auth/helpers";
import { environment } from "@/services/environment";
import {
  describeErrorEvent,
  isAuthenticationClose,
  logWebSocketClose,
  logWebSocketIssue,
} from "./websocket-logging";

interface Handlers {
  onConnect: () => void;
  onDisconnect: () => void;
  onMessage: (event: MessageEvent) => void;
}

export class BackendWebSocket {
  private socket: WebSocket | null = null;
  private authenticatedSocket: WebSocket | null = null;
  private connecting: Promise<void> | null = null;
  private resolveConnecting: (() => void) | null = null;
  private generation = 0;
  private stopped = false;
  private retryDelay = 1000;
  private retryTimer: ReturnType<typeof setTimeout> | undefined;
  private heartbeatTimer: ReturnType<typeof setInterval> | undefined;
  private heartbeatTimeout: ReturnType<typeof setTimeout> | undefined;

  constructor(
    private url: string,
    private handlers: Handlers,
  ) {}

  get connected() {
    return (
      !this.stopped &&
      this.socket !== null &&
      this.authenticatedSocket === this.socket &&
      this.socket.readyState === WebSocket.OPEN
    );
  }

  send(message: string): boolean {
    if (!this.socket || !this.connected) return false;
    this.socket.send(message);
    return true;
  }

  connect(): Promise<void> {
    if (hasDisconnectIntent() || this.connected) return Promise.resolve();
    if (this.connecting) return this.connecting;
    if (this.socket && !this.stopped) return this.pendingConnection();
    this.stopped = false;
    clearTimeout(this.retryTimer);
    const generation = ++this.generation;
    const pending = this.pendingConnection();
    void this.open(generation);
    return pending;
  }

  disconnect(): void {
    this.stopped = true;
    this.generation++;
    this.finishConnecting();
    const socket = this.socket;
    this.socket = null;
    this.authenticatedSocket = null;
    clearTimeout(this.retryTimer);
    this.stopHeartbeat();
    if (socket && socket.readyState < WebSocket.CLOSING) socket.close();
  }

  private async open(generation: number): Promise<void> {
    if (generation !== this.generation) return;
    if (this.stopped || hasDisconnectIntent()) {
      this.finishConnecting();
      return;
    }
    try {
      const { token, error } = await getWebSocketToken();
      if (generation !== this.generation) return;
      if (!token || error || hasDisconnectIntent()) {
        this.finishConnecting();
        return;
      }
      const socket = new WebSocket(
        `${this.url}?token=${encodeURIComponent(token)}`,
      );
      this.socket = socket;
      this.attach(socket, generation);
    } catch (error) {
      if (generation !== this.generation) return;
      this.finishConnecting();
      console.warn("[BackendAPI] Could not connect WebSocket:", error);
    }
  }

  private pendingConnection(): Promise<void> {
    return (this.connecting ??= new Promise((resolve) => {
      this.resolveConnecting = resolve;
    }));
  }

  private finishConnecting() {
    const resolve = this.resolveConnecting;
    this.resolveConnecting = null;
    this.connecting = null;
    resolve?.();
  }

  private current(socket: WebSocket, generation: number) {
    return this.socket === socket && this.generation === generation;
  }

  private attach(socket: WebSocket, generation: number) {
    socket.onopen = () => {
      if (!this.current(socket, generation) || hasDisconnectIntent()) {
        socket.close();
        return;
      }
      this.ping(socket, generation);
    };
    socket.onmessage = (event) => this.receive(socket, generation, event);
    socket.onclose = (event) => this.closed(socket, generation, event);
    socket.onerror = (event) => {
      if (
        !this.current(socket, generation) ||
        this.authenticatedSocket !== socket
      )
        return;
      const { summary, extra } = describeErrorEvent(event, this.url);
      logWebSocketIssue(
        "error",
        `[BackendAPI] WebSocket error: ${summary}`,
        extra,
      );
    };
  }

  private receive(socket: WebSocket, generation: number, event: MessageEvent) {
    if (!this.current(socket, generation)) return;
    if (hasDisconnectIntent()) {
      socket.close();
      return;
    }
    let message: unknown;
    try {
      message = JSON.parse(event.data);
    } catch {
      console.warn("[BackendAPI] Invalid WebSocket message");
      return;
    }
    if (!isHeartbeatPong(message)) {
      this.handlers.onMessage(event);
      return;
    }
    clearTimeout(this.heartbeatTimeout);
    if (this.authenticatedSocket === socket) return;
    // A pong follows backend authentication; transport open alone does not.
    this.authenticatedSocket = socket;
    this.retryDelay = 1000;
    this.heartbeatTimer = setInterval(
      () => this.ping(socket, generation),
      100_000,
    );
    this.finishConnecting();
    this.handlers.onConnect();
  }

  private closed(socket: WebSocket, generation: number, event: CloseEvent) {
    const connected = this.authenticatedSocket === socket;
    const intentional =
      this.stopped ||
      hasDisconnectIntent() ||
      !this.current(socket, generation);
    logWebSocketClose(event, this.url, connected, intentional);
    if (!this.current(socket, generation)) return;
    this.socket = null;
    this.authenticatedSocket = null;
    this.stopHeartbeat();
    const terminal = intentional || isAuthenticationClose(event.code);
    if (terminal) this.finishConnecting();
    else this.pendingConnection();
    if (!intentional && connected) this.handlers.onDisconnect();
    if (terminal || generation !== this.generation || this.stopped) return;
    if (hasDisconnectIntent()) {
      this.finishConnecting();
      return;
    }
    this.retryTimer = setTimeout(
      () => void this.open(generation),
      this.retryDelay,
    );
    this.retryDelay = Math.min(this.retryDelay * 2, 30_000);
  }

  private ping(socket: WebSocket, generation: number) {
    if (
      !this.current(socket, generation) ||
      socket.readyState !== WebSocket.OPEN
    )
      return;
    clearTimeout(this.heartbeatTimeout);
    this.heartbeatTimeout = setTimeout(() => {
      if (!this.current(socket, generation)) return;
      if (!hasDisconnectIntent())
        console.warn("[BackendAPI] WebSocket heartbeat timed out");
      socket.close();
    }, 10_000);
    socket.send(
      JSON.stringify({ method: "heartbeat", data: "ping", success: true }),
    );
  }

  private stopHeartbeat() {
    clearInterval(this.heartbeatTimer);
    clearTimeout(this.heartbeatTimeout);
  }
}

function hasDisconnectIntent(): boolean {
  if (!environment.isClientSide()) return false;
  try {
    return hasWebSocketDisconnectIntent();
  } catch {
    return false;
  }
}

function isHeartbeatPong(message: unknown): boolean {
  return (
    typeof message === "object" &&
    message !== null &&
    "method" in message &&
    message.method === "heartbeat" &&
    "data" in message &&
    message.data === "pong"
  );
}
