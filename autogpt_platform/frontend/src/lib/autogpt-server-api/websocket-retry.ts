import { logWebSocketIssue } from "./websocket-logging";

// A subscribe the server could not complete (Redis didn't confirm it in
// time, or its stream was lost) is sent again after this, then twice as long
// after each failure, so a short Redis hiccup doesn't cost a view its live
// updates until the next reconnect.
const FIRST_SUBSCRIBE_RETRY_MS = 1000;
const MAX_SUBSCRIBE_RETRIES = 5;

export interface WebSocketRequest {
  method: string;
  data?: unknown;
}

// ws_api.py's answer to a request it could not handle.
interface RequestFailure {
  error?: string;
  request: WebSocketRequest;
  retryable: boolean;
}

interface Retry {
  request: WebSocketRequest;
  sent: number;
  timer?: ReturnType<typeof setTimeout>;
}

/**
 * One connection's subscribes the server could not complete. Each is sent
 * again, with backoff, until it is confirmed, refused or out of retries.
 */
export class SubscribeRetries {
  private retries = new Map<string, Retry>();
  private reportedGivingUp = false;
  // A retry's send count when the ordering ping now out was sent: only a
  // retry sent before that ping is covered by its pong. A retry sent after
  // it — or sent again since — is not, so it stays pending until a later
  // ping covers it.
  private coveredByPing = new Map<string, number>();

  constructor(private resend: (request: WebSocketRequest) => void) {}

  /** Some subscribe waits to be sent again. */
  get waiting(): boolean {
    return [...this.retries.values()].some(
      (retry) => retry.timer !== undefined,
    );
  }

  /**
   * Reads a message from the server, acting on an error that names a
   * subscribe; true when that schedules a retry. An error that names no
   * request changes nothing.
   */
  read(message: unknown): boolean {
    if (!isSubscribeFailure(message)) return false;
    const key = `${message.request.method} ${sortedJSON(message.request.data)}`;
    const retry = this.retries.get(key) ?? {
      request: message.request,
      sent: 0,
    };
    // A refusal stands, even with another copy's retry already waiting.
    if (!message.retryable) {
      clearTimeout(retry.timer);
      this.retries.delete(key);
      this.logStopped(message, retry.sent);
      return false;
    }
    // Another view's copy of a subscribe already waiting to go again.
    if (retry.timer !== undefined) return false;
    if (retry.sent >= MAX_SUBSCRIBE_RETRIES) {
      this.retries.delete(key);
      this.logStopped(message, retry.sent);
      return false;
    }
    retry.timer = setTimeout(
      () => {
        retry.timer = undefined;
        retry.sent += 1;
        this.resend(retry.request);
      },
      withJitter(FIRST_SUBSCRIBE_RETRY_MS * 2 ** retry.sent),
    );
    this.retries.set(key, retry);
    return true;
  }

  /** A subscribe was sent again and waits for the server's answer. */
  get awaitingConfirmation(): boolean {
    return [...this.retries.values()].some(
      (retry) => retry.timer === undefined,
    );
  }

  /**
   * The ordering ping is out: the retries already sent are the ones its
   * pong vouches for. A retry sent after it is not covered, and a retry
   * sent again since has a newer send count, so neither is settled by it.
   */
  markPingCovered() {
    this.coveredByPing.clear();
    for (const [key, retry] of this.retries) {
      if (retry.timer === undefined) this.coveredByPing.set(key, retry.sent);
    }
  }

  /**
   * The server has answered everything sent before the ordering ping, so a
   * subscribe sent again that is covered by it and isn't waiting to go
   * again was confirmed. A retry the ping does not cover stays pending,
   * with its attempt count, until a later ping covers it.
   */
  settle() {
    for (const [key, retry] of this.retries) {
      if (
        retry.timer === undefined &&
        this.coveredByPing.get(key) === retry.sent
      ) {
        this.retries.delete(key);
      }
    }
    this.coveredByPing.clear();
  }

  clear() {
    this.retries.forEach((retry) => clearTimeout(retry.timer));
    this.retries.clear();
    this.coveredByPing.clear();
    this.reportedGivingUp = false;
  }

  private logStopped(failure: RequestFailure, retriesSent: number) {
    const subscribe = `[BackendAPI] WebSocket ${failure.request.method}`;
    if (!failure.retryable) {
      // A refusal stands (a cap, a run that is gone), and the server logs it.
      console.info(
        `${subscribe} refused: ${failure.error ?? "no reason given"}`,
      );
      return;
    }
    const givingUp = `${subscribe} still failing after ${retriesSent} retries; its live updates stop until the next reconnect`;
    // A warning is a Sentry event, and one Redis outage can fail every
    // subscribe of a connection: report the first, keep the rest as crumbs.
    if (this.reportedGivingUp) {
      console.info(givingUp);
      return;
    }
    this.reportedGivingUp = true;
    logWebSocketIssue("warn", givingUp, {
      ws_request: failure.request,
      ws_error: failure.error ?? null,
    });
  }
}

// Sorted keys: the server echoes the request's data, and its key order is
// not a promise, so two echoes of one subscribe still make one key.
function sortedJSON(value: unknown) {
  return JSON.stringify(value, (_key, nested: unknown) =>
    nested && typeof nested === "object" && !Array.isArray(nested)
      ? Object.fromEntries(
          Object.entries(nested).sort(([a], [b]) => (a < b ? -1 : 1)),
        )
      : nested,
  );
}

// Equal jitter: a random point in the second half of the step, so clients
// dropped together (a server restart) don't all come back at once.
export function withJitter(delay: number) {
  return Math.round(delay / 2 + (Math.random() * delay) / 2);
}

function isSubscribeFailure(message: unknown): message is RequestFailure {
  if (typeof message !== "object" || message === null) return false;
  if (!("method" in message) || message.method !== "error") return false;
  if (!("retryable" in message) || typeof message.retryable !== "boolean") {
    return false;
  }
  if (!("request" in message)) return false;
  const request = message.request;
  return (
    typeof request === "object" &&
    request !== null &&
    "method" in request &&
    typeof request.method === "string" &&
    request.method.startsWith("subscribe_")
  );
}
