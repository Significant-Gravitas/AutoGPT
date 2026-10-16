import * as Sentry from "@sentry/nextjs";

import { readSseFrames } from "./sseClient";
import { applyEntry, jsonEqual, type StreamEntry } from "./turnConverter";
import {
  emptyTurnLog,
  logRowFromPersisted,
  parseJsonLoose,
  type LogRow,
  type PersistedRow,
  type TurnLog,
} from "./turnLog";

/**
 * Shadow mode: the stream the AI SDK renders is also folded by our converter,
 * which is checked against the server at every checkpoint and against the
 * persisted rows once the turn is over. It renders nothing and reports drift
 * to Sentry; any failure in it degrades to silence.
 */
interface ShadowTurn {
  log: TurnLog;
  verified: number;
  reportedErrors: number;
  compared: boolean;
  /** Its stream ended before `finish`: closed from outside, or cancelled by the client. */
  ended: "abandoned" | "stopped" | null;
}

/** A finished turn keeps its slot until compared, so the next turn's entries
 *  cannot take it before the session view that checks it arrives. */
interface SessionShadow {
  current: ShadowTurn;
  previous: ShadowTurn | null;
}

// Each slot holds a whole turn's rows; only the most recent sessions keep one.
const MAX_SESSIONS = 5;
const shadows = new Map<string, SessionShadow>();
let shadowRate = 0;
const sampledSessions = new Map<string, boolean>();

/** The share of chat sessions the shadow tees, from `copilot-stream-shadow`; 0 is off. */
export function setStreamShadowRate(rate: unknown) {
  shadowRate =
    typeof rate === "number" && Number.isFinite(rate)
      ? Math.min(1, Math.max(0, rate))
      : 0;
}

/** Wrap the transport's fetch so a sampled session's stream responses are teed into the shadow. */
export function createShadowFetch(
  sessionId: string,
  inner: typeof fetch = (input, init) => fetch(input, init),
): typeof fetch {
  return async (input, init) => {
    const response = await inner(input, init);
    if (!response.ok || !response.body || !isSampled(sessionId)) {
      return response;
    }
    const [main, tap] = response.body.tee();
    const stopTap = new AbortController();
    const stop = () => stopTap.abort();
    init?.signal?.addEventListener("abort", stop);
    let turnId: string | null = null;
    void readSseFrames(
      tap,
      (frame) => {
        if (frame.kind !== "entry") return;
        turnId = frame.entry.turn;
        applyShadowEntry(sessionId, frame.entry);
      },
      stopTap.signal,
    )
      .catch(() => {})
      .finally(() => {
        init?.signal?.removeEventListener("abort", stop);
        markEnded(sessionId, turnId, stopTap.signal.aborted);
      });
    return new Response(
      onCancel(main, () => stopTap.abort()),
      {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      },
    );
  };
}

export function applyShadowEntry(sessionId: string, entry: StreamEntry) {
  try {
    const shadow = shadowFor(sessionId, entry.turn);
    shadow.log = applyEntry(shadow.log, entry);
    shadow.ended = null;
    if (!isWholeTurn(shadow.log)) return;
    reportProtocolErrors(shadow);
    void verifyCheckpoints(shadow);
  } catch {
    // The shadow must never break the chat it watches.
  }
}

/** Once the DB view no longer runs a shadowed finished turn, diff its rows. */
export function compareShadowWithSession(
  sessionId: string,
  session: SessionView,
) {
  const slot = shadows.get(sessionId);
  if (!slot) return;
  if (slot.previous) compareTurn(slot.previous, session);
  if (slot.previous?.compared) slot.previous = null;
  compareTurn(slot.current, session);
}

export function getShadowLog(sessionId: string): TurnLog | null {
  return shadows.get(sessionId)?.current.log ?? null;
}

export function resetShadows() {
  shadows.clear();
  sampledSessions.clear();
  shadowRate = 0;
}

/** One entry per differing row, with the fields that differ. */
export function diffRows(
  live: readonly LogRow[],
  persisted: readonly LogRow[],
) {
  const diffs: { index: number; fields: string[] }[] = [];
  for (let i = 0; i < Math.max(live.length, persisted.length); i++) {
    const fields = rowDifferences(live[i], persisted[i]);
    if (fields.length > 0) diffs.push({ index: i, fields });
  }
  return diffs;
}

interface SessionView {
  messages?: readonly unknown[] | null;
  active_stream?: { turn_id: string } | null;
  has_more_messages?: boolean;
}

// Sampled per session, not per request, so a turn's resumes are teed when its POST was.
function isSampled(sessionId: string) {
  if (shadowRate <= 0) return false;
  let sampled = sampledSessions.get(sessionId);
  if (sampled === undefined) {
    sampled = Math.random() < shadowRate;
    sampledSessions.set(sessionId, sampled);
  }
  return sampled;
}

function markEnded(
  sessionId: string,
  turnId: string | null,
  byClient: boolean,
) {
  const slot = shadows.get(sessionId);
  const shadow = [slot?.current, slot?.previous].find(
    (s) => s && turnId !== null && s.log.turnId === turnId,
  );
  const over =
    shadow?.log.status === "finished" || shadow?.log.status === "failed";
  if (!shadow || over) return;
  shadow.ended = byClient ? "stopped" : "abandoned";
}

function shadowFor(sessionId: string, turnId: string): ShadowTurn {
  const slot = shadows.get(sessionId);
  if (slot?.current.log.turnId === turnId) return slot.current;
  if (slot?.previous?.log.turnId === turnId) return slot.previous;
  const current = {
    log: emptyTurnLog(),
    verified: 0,
    reportedErrors: 0,
    compared: false,
    ended: null,
  };
  const pending = slot && !slot.current.compared ? slot.current : null;
  shadows.delete(sessionId);
  shadows.set(sessionId, {
    current,
    previous: pending ?? slot?.previous ?? null,
  });
  const oldest = shadows.keys().next().value;
  if (shadows.size > MAX_SESSIONS && oldest !== undefined) {
    shadows.delete(oldest);
  }
  return current;
}

function compareTurn(shadow: ShadowTurn, session: SessionView) {
  if (shadow.compared || !isWholeTurn(shadow.log)) return;
  const { log } = shadow;
  if (session.active_stream?.turn_id === log.turnId) return;
  if (log.status !== "finished" && log.status !== "failed") {
    // The server is done with a turn whose stream the client lost: what the
    // screen shows stopped where the stream did.
    if (!shadow.ended) return;
    shadow.compared = true;
    reportDrift(shadow.ended, log, {
      rows: log.rows.length,
      checkpoints: log.checkpoints.length,
    });
    return;
  }
  const last = log.checkpoints[log.checkpoints.length - 1];
  if (!last) {
    shadow.compared = true;
    reportDrift("finish_without_checkpoint", log, { rows: log.rows.length });
    return;
  }
  const numbered = (session.messages ?? [])
    .map((row) => row as PersistedRow)
    .filter((row) => typeof row.sequence === "number")
    .sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0));
  // The session GET returns a window; a turn that starts before it is not checked.
  const windowStart = numbered[0]?.sequence ?? Infinity;
  if (session.has_more_messages !== false && windowStart > last.sequence)
    return;
  shadow.compared = true;
  const turnRows = numbered
    .filter((row) => (row.sequence ?? -1) >= last.sequence)
    .slice(0, last.rows)
    .map(logRowFromPersisted);
  const diffs = diffRows(log.rows, turnRows);
  if (diffs.length > 0) reportDrift("finish", log, { diffs });
}

// A replay that opens past the turn's `start` (a trimmed stream) is a tail,
// not a turn; comparing it with the whole turn would report drift that is not.
function isWholeTurn(log: TurnLog) {
  return log.messageId !== null;
}

function reportProtocolErrors(shadow: ShadowTurn) {
  const errors = shadow.log.protocolErrors.slice(shadow.reportedErrors);
  shadow.reportedErrors = shadow.log.protocolErrors.length;
  for (const error of errors) {
    reportDrift("protocol_error", shadow.log, { ...error });
  }
}

async function verifyCheckpoints(shadow: ShadowTurn) {
  while (shadow.verified < shadow.log.checkpoints.length) {
    const checkpoint = shadow.log.checkpoints[shadow.verified++];
    const digest = await sha256Hex(checkpoint.canonical);
    if (digest !== null && digest !== checkpoint.digest) {
      reportDrift("checkpoint", shadow.log, {
        entryId: checkpoint.entryId,
        rows: checkpoint.rows,
        serverDigest: checkpoint.digest,
        clientDigest: digest,
      });
    }
  }
}

function rowDifferences(a: LogRow | undefined, b: LogRow | undefined) {
  if (!a || !b) return ["missing"];
  const fields: string[] = [];
  if (a.role !== b.role) fields.push("role");
  if (a.role === "tool") {
    if (a.toolCallId !== b.toolCallId) fields.push("tool_call_id");
    if (!jsonEqual(parseJsonLoose(a.content), parseJsonLoose(b.content))) {
      fields.push("output");
    }
  } else if (a.content !== b.content) {
    fields.push("content");
  }
  const sameCalls =
    a.toolCalls.length === b.toolCalls.length &&
    a.toolCalls.every((call, i) => {
      const other = b.toolCalls[i];
      return (
        call.id === other.id &&
        call.name === other.name &&
        jsonEqual(call.input, other.input)
      );
    });
  if (!sameCalls) fields.push("tool_calls");
  return fields;
}

function reportDrift(
  kind: string,
  log: TurnLog,
  extra: Record<string, unknown>,
) {
  Sentry.captureMessage(`copilot stream drift: ${kind}`, {
    level: "warning",
    tags: { stream_drift: kind },
    extra: { turnId: log.turnId, cursor: log.cursor, ...extra },
  });
}

// `crypto.subtle` exists only in secure contexts; a plain-HTTP LAN origin skips the check.
async function sha256Hex(text: string): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const hash = await subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(hash), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
}

// A tee branch outlives its sibling: when the SDK cancels the body, the tap
// must stop too or it holds the connection open.
function onCancel(stream: ReadableStream<Uint8Array>, cancelled: () => void) {
  const reader = stream.getReader();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) controller.close();
      else controller.enqueue(value);
    },
    cancel(reason) {
      cancelled();
      return reader.cancel(reason);
    },
  });
}
