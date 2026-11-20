import { getWebSocketToken } from "@/lib/auth/actions";
import { hasWebSocketDisconnectIntent } from "@/lib/auth/helpers";
import { environment } from "@/services/environment";
import {
  describeErrorEvent,
  isAuthenticationClose,
  logWebSocketClose,
  logWebSocketIssue,
} from "./websocket-logging";
import {
  SubscribeRetries,
  type WebSocketRequest,
  withJitter,
} from "./websocket-retry";

interface Handlers {
  onConnect: () => void;
  /**
   * Whenever live state should be re-read: after a reconnect, once the
   * server has handled what onConnect sent, and after a subscribe it could
   * not complete settles on a retry, on any connection.
   */
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
// The server answers the ordering ping only after every resubscription, and
// opening a shard's event stream for one can take it up to 15 s (OPEN_TIMEOUT_S
// in backend/data/sharded_pubsub.py); each answer to a subscribe restarts this.
const RESUBSCRIBED_TIMEOUT_MS = 30_000;
// After a mass disconnect (a deploy) every tab reconnects in the same retry
// window, so each waits a random part of this before its views refetch.
const RECONNECT_REFETCH_SPREAD_MS = 2000;
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
  private resubscribedTimer: ReturnType<typeof setTimeout> | undefined;
  private heartbeatTimer: ReturnType<typeof setInterval> | undefined;
  private heartbeatTimeout: ReturnType<typeof setTimeout> | undefined;
  private pongsOwed = 0;
  private pongsUntilResubscribed = 0;
  private sentDuringResubscribeCheck = false;
  // What makes the views re-read at the next caught-up pong.
  private rereadAfterReconnect = false;
  private rereadAfterRetries = false;
  private watchingPage = false;
  private retries = new SubscribeRetries((request) =>
    this.resubscribe(request),
  );

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
    // The pong to an ordering ping already sent can't vouch for this.
    if (this.pongsUntilResubscribed > 0) this.sentDuringResubscribeCheck = true;
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
    // Signing out ends the session: the next connection is a first one.
    this.retryDelay = FIRST_RETRY_DELAY_MS;
    this.hasConnected = false;
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
      // The server is still working through the resubscriptions.
      if (this.pongsUntilResubscribed > 0 && answersARequest(message)) {
        this.armDeadline(socket, generation, RESUBSCRIBED_TIMEOUT_MS);
      }
      if (this.retries.read(message)) this.rereadAfterRetries = true;
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
      if (this.sentDuringResubscribeCheck) {
        // A view subscribed after the ping: ask again, so no view re-reads
        // before the server has handled its subscription.
        this.checkResubscribed(socket, generation);
        return;
      }
      this.caughtUp();
    }
  }

  // The server has handled everything sent before the last ping. After a
  // reconnect the views re-read at once: the subscriptions that went through
  // are live. A subscribe still being retried makes them re-read once more
  // when it settles, so the last read follows it.
  private caughtUp() {
    this.retries.settle();
    const retrying = this.retries.waiting;
    if (this.rereadAfterReconnect) {
      // A reconnect only counts as healthy once it has caught up.
      this.startHealthyTimer();
    } else if (!this.rereadAfterRetries || retrying) {
      return;
    }
    this.rereadAfterReconnect = false;
    this.rereadAfterRetries = retrying;
    this.handlers.onReconnect();
  }

  // A pong follows backend authentication; transport open alone does not.
  private authenticated(socket: WebSocket, generation: number) {
    this.authenticatedSocket = socket;
    this.heartbeatTimer = setInterval(() => {
      // One check at a time: another ping would push its deadline back.
      if (this.pongsOwed === 0) this.ping(socket, generation);
    }, HEARTBEAT_INTERVAL_MS);
    const reconnected = this.hasConnected;
    this.hasConnected = true;
    this.finishConnecting();
    this.handlers.onConnect();
    if (!reconnected) {
      this.startHealthyTimer();
      return;
    }
    this.rereadAfterReconnect = true;
    this.resubscribedTimer = setTimeout(() => {
      this.resubscribedTimer = undefined;
      this.checkResubscribed(socket, generation);
    }, Math.random() * RECONNECT_REFETCH_SPREAD_MS);
  }

  // The server answers a socket's messages in order, so the pong to this
  // ping means it has handled every subscription sent before it: the ones
  // onConnect sent again, and any a view added during the spread.
  private checkResubscribed(socket: WebSocket, generation: number) {
    if (!this.current(socket, generation) || !this.connected) return;
    this.sentDuringResubscribeCheck = false;
    this.ping(socket, generation, RESUBSCRIBED_TIMEOUT_MS);
    this.pongsUntilResubscribed = this.pongsOwed;
  }

  // Views re-read once the server has answered this, so a ping follows it,
  // unless one already out or the one after the spread will.
  private resubscribe(request: WebSocketRequest) {
    const socket = this.socket;
    // Not sent: the socket is closing, and its close clears the retries.
    if (!socket || !this.send(JSON.stringify(request))) return;
    if (this.pongsUntilResubscribed > 0) return;
    if (this.resubscribedTimer !== undefined) return;
    this.checkResubscribed(socket, this.generation);
  }

  private startHealthyTimer() {
    clearTimeout(this.healthyTimer);
    this.healthyTimer = setTimeout(() => {
      this.retryDelay = FIRST_RETRY_DELAY_MS;
    }, HEALTHY_AFTER_MS);
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

  private ping(
    socket: WebSocket,
    generation: number,
    timeoutMs = HEARTBEAT_TIMEOUT_MS,
  ) {
    if (
      !this.current(socket, generation) ||
      socket.readyState !== WebSocket.OPEN
    )
      return;
    this.armDeadline(socket, generation, timeoutMs);
    this.pongsOwed += 1;
    socket.send(PING);
  }

  private armDeadline(socket: WebSocket, generation: number, ms: number) {
    clearTimeout(this.heartbeatTimeout);
    this.heartbeatTimeout = setTimeout(() => {
      if (!this.current(socket, generation)) return;
      if (!hasDisconnectIntent())
        console.warn("[BackendAPI] WebSocket heartbeat timed out");
      socket.close();
    }, ms);
  }

  private stopHealthChecks() {
    clearInterval(this.heartbeatTimer);
    clearTimeout(this.heartbeatTimeout);
    clearTimeout(this.healthyTimer);
    clearTimeout(this.resubscribedTimer);
    this.resubscribedTimer = undefined;
    this.retries.clear();
    this.rereadAfterReconnect = false;
    this.rereadAfterRetries = false;
    this.pongsOwed = 0;
    this.pongsUntilResubscribed = 0;
    this.sentDuringResubscribeCheck = false;
  }

  private watchPage() {
    if (this.watchingPage || !environment.isClientSide()) return;
    this.watchingPage = true;
    window.addEventListener("online", this.checkNow);
    window.addEventListener("focus", this.checkNow);
    document.addEventListener("visibilitychange", this.checkWhenVisible);
  }

  private unwatchPage() {
    if (!this.watchingPage) return;
    this.watchingPage = false;
    window.removeEventListener("online", this.checkNow);
    window.removeEventListener("focus", this.checkNow);
    document.removeEventListener("visibilitychange", this.checkWhenVisible);
  }

  private checkWhenVisible = () => {
    if (document.visibilityState === "visible") this.checkNow();
  };

  // Back online, in view or in focus: a socket that seems open may be dead
  // (a laptop that slept), so make it answer within the heartbeat timeout,
  // and a pending retry runs now instead of waiting out the backoff.
  private checkNow = () => {
    if (this.stopped || hasDisconnectIntent()) return;
    if (this.connected && this.socket) {
      // One check at a time: another ping would push its deadline back.
      if (this.pongsOwed === 0) this.ping(this.socket, this.generation);
    } else if (this.retryTimer !== undefined) {
      this.cancelRetry();
      void this.open(this.generation);
    }
  };
}

function hasDisconnectIntent(): boolean {
  if (!environment.isClientSide()) return false;
  try {
    return hasWebSocketDisconnectIntent();
  } catch {
    return false;
  }
}

// The server's answer to a subscribe: an acknowledgement or an error.
function answersARequest(message: unknown): boolean {
  if (typeof message !== "object" || message === null) return false;
  const method = "method" in message ? message.method : undefined;
  return (
    method === "error" ||
    (typeof method === "string" && method.startsWith("subscribe_"))
  );
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
