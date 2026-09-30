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
}

const shadows = new Map<string, ShadowTurn>();

/** Wrap the transport's fetch so every stream response is teed into the shadow. */
export function createShadowFetch(
  sessionId: string,
  inner: typeof fetch = (input, init) => fetch(input, init),
): typeof fetch {
  return async (input, init) => {
    const response = await inner(input, init);
    if (!response.ok || !response.body) return response;
    const [main, tap] = response.body.tee();
    const stopTap = new AbortController();
    const stop = () => stopTap.abort();
    init?.signal?.addEventListener("abort", stop);
    void readSseFrames(
      tap,
      (frame) => {
        if (frame.kind === "entry") applyShadowEntry(sessionId, frame.entry);
      },
      stopTap.signal,
    )
      .catch(() => {})
      .finally(() => init?.signal?.removeEventListener("abort", stop));
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
    let shadow = shadows.get(sessionId);
    if (!shadow || shadow.log.turnId !== entry.turn) {
      shadow = {
        log: emptyTurnLog(),
        verified: 0,
        reportedErrors: 0,
        compared: false,
      };
      shadows.set(sessionId, shadow);
    }
    shadow.log = applyEntry(shadow.log, entry);
    if (!isWholeTurn(shadow.log)) return;
    reportProtocolErrors(shadow);
    void verifyCheckpoints(shadow);
  } catch {
    // The shadow must never break the chat it watches.
  }
}

/** Once the DB view no longer runs the shadow's finished turn, diff its rows. */
export function compareShadowWithSession(
  sessionId: string,
  session: {
    messages?: readonly unknown[] | null;
    active_stream?: { turn_id: string } | null;
    has_more_messages?: boolean;
  },
) {
  const shadow = shadows.get(sessionId);
  if (!shadow || shadow.compared || !isWholeTurn(shadow.log)) return;
  const { log } = shadow;
  if (log.status !== "finished" && log.status !== "failed") return;
  if (session.active_stream?.turn_id === log.turnId) return;
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

export function getShadowLog(sessionId: string): TurnLog | null {
  return shadows.get(sessionId)?.log ?? null;
}

export function resetShadows() {
  shadows.clear();
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
