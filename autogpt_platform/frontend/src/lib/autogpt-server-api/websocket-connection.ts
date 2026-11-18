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
  /** After a reconnect, once the server has handled what onConnect sent. */
  onReconnect: () => void;
  onDisconnect: () => void;
  onMessage: (event: MessageEvent) => void;
}

const FIRST_RETRY_DELAY_MS = 1000;
const MAX_RETRY_DELAY_MS = 30_000;
// The server can accept a socket and close it again at once (its event
// stream keeps failing), so only a connection that stays up this long
// resets the backoff.
const HEALTHY_AFTER_MS = 10_000;
const HEARTBEAT_INTERVAL_MS = 100_000;
const HEARTBEAT_TIMEOUT_MS = 10_000;
const PING = JSON.stringify({
  method: "heartbeat",
  data: "ping",
  success: true,
});

export class BackendWebSocket {
  private socket: WebSocket | null = null;
  private authenticatedSocket: WebSocket | null = null;
  private connecting: Promise<void> | null = null;
  private resolveConnecting: (() => void) | null = null;
  private generation = 0;
  private stopped = false;
  private hasConnected = false;
  private retryDelay = FIRST_RETRY_DELAY_MS;
  private retryTimer: ReturnType<typeof setTimeout> | undefined;
  private healthyTimer: ReturnType<typeof setTimeout> | undefined;
  private heartbeatTimer: ReturnType<typeof setInterval> | undefined;
  private heartbeatTimeout: ReturnType<typeof setTimeout> | undefined;
  private pongsOwed = 0;
  private pongsUntilResubscribed = 0;
  private watchingPage = false;

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
    this.watchPage();
    if (this.connecting) return this.connecting;
    if (this.socket && !this.stopped) return this.pendingConnection();
    this.stopped = false;
    this.cancelRetry();
    const generation = ++this.generation;
    const pending = this.pendingConnection();
    void this.open(generation);
    return pending;
  }

  disconnect(): void {
    this.stopped = true;
    this.generation++;
    this.finishConnecting();
    this.unwatchPage();
    const socket = this.socket;
    this.socket = null;
    this.authenticatedSocket = null;
    this.cancelRetry();
    this.stopHealthChecks();
    if (socket && socket.readyState < WebSocket.CLOSING) socket.close();
  }

  private async open(generation: number): Promise<void> {
    if (generation !== this.generation) return;
    if (this.stopped || hasDisconnectIntent()) {
      this.finishConnecting();
      return;
    }
    try {
      const { token, error, retryable } = await getWebSocketToken();
      if (generation !== this.generation) return;
      if (hasDisconnectIntent()) {
        this.finishConnecting();
        return;
      }
      if (!token || error) {
        if (retryable) this.scheduleRetry(generation);
        else this.finishConnecting();
        return;
      }
      const socket = new WebSocket(
        `${this.url}?token=${encodeURIComponent(token)}`,
      );
      this.socket = socket;
      this.attach(socket, generation);
    } catch (error) {
      if (generation !== this.generation) return;
      console.warn("[BackendAPI] Could not connect WebSocket:", error);
      this.scheduleRetry(generation);
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
    this.pongsOwed = Math.max(0, this.pongsOwed - 1);
    if (this.pongsOwed === 0) clearTimeout(this.heartbeatTimeout);
    if (this.authenticatedSocket !== socket) {
      this.authenticated(socket, generation);
    } else if (
      this.pongsUntilResubscribed > 0 &&
      --this.pongsUntilResubscribed === 0
    ) {
      this.handlers.onReconnect();
    }
  }

  // A pong follows backend authentication; transport open alone does not.
  private authenticated(socket: WebSocket, generation: number) {
    this.authenticatedSocket = socket;
    this.healthyTimer = setTimeout(() => {
      this.retryDelay = FIRST_RETRY_DELAY_MS;
    }, HEALTHY_AFTER_MS);
    this.heartbeatTimer = setInterval(
      () => this.ping(socket, generation),
      HEARTBEAT_INTERVAL_MS,
    );
    const reconnected = this.hasConnected;
    this.hasConnected = true;
    this.finishConnecting();
    this.handlers.onConnect();
    if (!reconnected || !this.current(socket, generation)) return;
    // The server answers a socket's messages in order, so the pong to this
    // ping means it has handled every subscription onConnect sent again.
    this.pongsOwed += 1;
    this.pongsUntilResubscribed = this.pongsOwed;
    socket.send(PING);
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
    this.stopHealthChecks();
    const terminal = intentional || isAuthenticationClose(event.code);
    if (terminal) this.finishConnecting();
    else this.pendingConnection();
    if (!intentional && connected) this.handlers.onDisconnect();
    if (terminal || generation !== this.generation || this.stopped) return;
    this.scheduleRetry(generation);
  }

  private scheduleRetry(generation: number) {
    if (generation !== this.generation || this.stopped) return;
    if (hasDisconnectIntent()) {
      this.finishConnecting();
      return;
    }
    this.pendingConnection();
    this.cancelRetry();
    this.retryTimer = setTimeout(() => {
      this.retryTimer = undefined;
      void this.open(generation);
    }, withJitter(this.retryDelay));
    this.retryDelay = Math.min(this.retryDelay * 2, MAX_RETRY_DELAY_MS);
  }

  private cancelRetry() {
    clearTimeout(this.retryTimer);
    this.retryTimer = undefined;
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
    }, HEARTBEAT_TIMEOUT_MS);
    this.pongsOwed += 1;
    socket.send(PING);
  }

  private stopHealthChecks() {
    clearInterval(this.heartbeatTimer);
    clearTimeout(this.heartbeatTimeout);
    clearTimeout(this.healthyTimer);
    this.pongsOwed = 0;
    this.pongsUntilResubscribed = 0;
  }

  private watchPage() {
    if (this.watchingPage || !environment.isClientSide()) return;
    this.watchingPage = true;
    window.addEventListener("online", this.checkNow);
    document.addEventListener("visibilitychange", this.checkWhenVisible);
  }

  private unwatchPage() {
    if (!this.watchingPage) return;
    this.watchingPage = false;
    window.removeEventListener("online", this.checkNow);
    document.removeEventListener("visibilitychange", this.checkWhenVisible);
  }

  private checkWhenVisible = () => {
    if (document.visibilityState === "visible") this.checkNow();
  };

  // Back online or back in view: a socket that seems open may be dead
  // (a laptop that slept), so make it answer within the heartbeat timeout,
  // and a pending retry runs now instead of waiting out the backoff.
  private checkNow = () => {
    if (this.stopped || hasDisconnectIntent()) return;
    if (this.connected && this.socket) {
      this.ping(this.socket, this.generation);
    } else if (this.retryTimer !== undefined) {
      this.cancelRetry();
      void this.open(this.generation);
    }
  };
}

// Equal jitter: a random point in the second half of the step, so clients
// dropped together (a server restart) don't all come back at once.
function withJitter(delay: number) {
  return Math.round(delay / 2 + (Math.random() * delay) / 2);
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
