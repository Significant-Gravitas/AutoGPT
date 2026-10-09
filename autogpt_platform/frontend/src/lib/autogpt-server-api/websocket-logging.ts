import * as Sentry from "@sentry/nextjs";

/**
 * `captureConsoleIntegration` (see `instrumentation-client.ts`) turns every
 * console.error/warn into a Sentry event whose title is the console arguments
 * `String()`-ed and joined with a space, so any object argument becomes
 * "[object …]". These helpers flatten WebSocket events into a readable summary,
 * which is logged as the only console argument, plus structured Sentry extras.
 */

// RFC 6455 status codes, plus the 4xxx codes the backend sends itself
// (see autogpt_platform/backend/backend/api/ws_api.py).
const CLOSE_CODE_NAMES: Record<number, string> = {
  1000: "normal closure",
  1001: "going away",
  1002: "protocol error",
  1003: "unsupported data",
  1005: "no status received",
  1006: "abnormal closure",
  1007: "invalid frame payload data",
  1008: "policy violation",
  1009: "message too big",
  1010: "mandatory extension missing",
  1011: "internal server error",
  1012: "service restart",
  1013: "try again later",
  1014: "bad gateway",
  1015: "TLS handshake failure",
  4001: "missing authentication token",
  4002: "invalid token",
  4003: "invalid token",
};

const READY_STATE_NAMES: Record<number, string> = {
  0: "CONNECTING",
  1: "OPEN",
  2: "CLOSING",
  3: "CLOSED",
};

export type WebSocketEventDetails = {
  /** Goes into the log line, and so into the Sentry issue title. */
  summary: string;
  /** Goes into the Sentry event's extras. */
  extra: Record<string, unknown>;
};

export function describeCloseEvent(
  event: Pick<CloseEvent, "code" | "reason" | "wasClean">,
  url: string,
  phase: "connecting" | "connected",
): WebSocketEventDetails {
  const name = CLOSE_CODE_NAMES[event.code] ?? "unknown code";
  const reason = event.reason || "(none)";
  return {
    summary: `code ${event.code} (${name}), reason "${reason}", wasClean ${Boolean(
      event.wasClean,
    )}`,
    extra: {
      ws_close_code: event.code,
      ws_close_code_name: name,
      ws_close_reason: event.reason || "",
      ws_was_clean: Boolean(event.wasClean),
      ws_phase: phase,
      ws_url: stripQuery(url),
    },
  };
}

export function describeErrorEvent(
  event: Pick<Event, "type"> & { target?: unknown },
  url: string,
): WebSocketEventDetails {
  // The spec deliberately gives error events no detail, so the socket's own
  // state is all there is to report beyond the event type.
  const target = event.target as { readyState?: number; url?: string } | null;
  const readyState = target?.readyState;
  const readyStateName =
    readyState !== undefined
      ? (READY_STATE_NAMES[readyState] ?? "unknown")
      : "unknown";
  const targetUrl = stripQuery(target?.url ?? url);
  return {
    summary: `type "${event.type || "error"}", readyState ${readyStateName}, url ${targetUrl}`,
    extra: {
      ws_event_type: event.type || "error",
      ws_ready_state: readyState ?? null,
      ws_ready_state_name: readyStateName,
      ws_url: targetUrl,
    },
  };
}

/**
 * Logs `message` and attaches `extra` to the Sentry event that
 * `captureConsoleIntegration` creates from that same console call: the
 * integration captures on a fork of the scope that is current while console
 * runs, so extras set here are inherited by the event.
 */
export function logWebSocketIssue(
  level: "error" | "warn",
  message: string,
  extra: Record<string, unknown>,
): void {
  Sentry.withScope((scope) => {
    scope.setExtras(extra);
    console[level](message);
  });
}

/** Keeps the auth token out of anything we report. */
function stripQuery(url: string): string {
  const queryStart = url.indexOf("?");
  return queryStart === -1 ? url : url.slice(0, queryStart);
}
