import type { UIMessage } from "ai";

import { CANCELLED_MARKER } from "../useCopilotStop";
import {
  COPILOT_RETRYABLE_ERROR_PREFIX,
  isMarker,
  logRowFromPersisted,
  type LogRow,
  type PersistedRow,
  type TurnLog,
} from "./turnLog";
import { diffRows } from "./turnShadow";

/**
 * The runtime's tail: the prompts, turns and persisted rows it renders after
 * the session's history, and which of the session's persisted rows belong to
 * each. Everything here is pure; the runtime holds the state.
 */

export interface UserSegment {
  kind: "user";
  key: string;
  message: UIMessage;
  sequence: number | null;
  /** `sent` is the turn's own prompt; `chip` a promoted follow-up. */
  origin: "sent" | "chip";
  rawId: string | null;
  createdAt: string | null;
}

export interface TurnSegment {
  kind: "turn";
  key: string;
  log: TurnLog;
  ended: boolean;
  stopped: boolean;
  /** Two drift events on one turn: the persisted view is shown instead. */
  frozen: boolean;
  reconciled: boolean;
  /** Fields that never stream, adopted at the reconcile. */
  durationMs: number | null;
  createdAt: string | null;
  /** When each persisted row was written, by row key. */
  rowCreatedAt: Readonly<Record<string, string>>;
  /** Where an attached turn's rows begin before it names a checkpoint. */
  startHint: number | null;
}

/** Persisted rows the tail holds but no turn of this mount produced. */
export interface RowsSegment {
  kind: "rows";
  key: string;
  rows: readonly PersistedRow[];
}

export type Segment = UserSegment | TurnSegment | RowsSegment;

export interface TurnCheckpoint {
  entry_id: string;
  rows: number;
  sequence: number;
}

export const INTERRUPTED_MARKER = `${COPILOT_RETRYABLE_ERROR_PREFIX} Response was interrupted. Resend to try again.`;
export const FROZEN_NOTE =
  "[__COPILOT_SYSTEM_e3b0__] This reply is shown as saved; it will update when the turn ends.";

export function newTurnSegment(
  turnId: string,
  log: TurnLog,
  startHint: number | null = null,
): TurnSegment {
  return {
    kind: "turn",
    key: `turn:${turnId}`,
    log,
    ended: false,
    stopped: false,
    frozen: false,
    reconciled: false,
    durationMs: null,
    createdAt: null,
    rowCreatedAt: {},
    startHint,
  };
}

export function userSegment(
  message: UIMessage,
  origin: UserSegment["origin"],
): UserSegment {
  return {
    kind: "user",
    key: message.id,
    message,
    sequence: null,
    origin,
    rawId: null,
    createdAt: null,
  };
}

/** Where a turn's rows start: its first checkpoint, its seed, or the row after its prompt. */
export function turnStart(
  segments: readonly Segment[],
  ownedFrom: number | null,
  seg: TurnSegment,
  rows: readonly PersistedRow[],
) {
  const known = segmentStart(seg);
  if (known !== null) return known;
  const index = segments.indexOf(seg);
  const before = segments[index - 1];
  if (before?.kind === "user") {
    const prompt =
      before.sequence ?? locatePrompt(before, rows, ownedFrom)?.sequence;
    return typeof prompt === "number" ? prompt + 1 : null;
  }
  return index === 0 ? ownedFrom : null;
}

/**
 * A turn's persisted rows: from its start up to the next segment, whose
 * prompt is found by its text while it has no sequence of its own yet (a
 * send right after a stop).
 */
export function turnRows(
  segments: readonly Segment[],
  ownedFrom: number | null,
  seg: TurnSegment,
  rows: readonly PersistedRow[],
) {
  const start = turnStart(segments, ownedFrom, seg, rows);
  if (start === null) return null;
  const end =
    segments
      .slice(segments.indexOf(seg) + 1)
      .map((later) => {
        const known = segmentStart(later);
        if (known !== null || later.kind !== "user" || later.origin !== "sent")
          return known;
        return locatePrompt(later, rows, start + 1)?.sequence ?? null;
      })
      .find((n): n is number => typeof n === "number") ?? Infinity;
  const raw = rows.filter(
    (r) =>
      typeof r.sequence === "number" && r.sequence >= start && r.sequence < end,
  );
  return { start, raw, rows: raw.map(logRowFromPersisted) };
}

/**
 * The end-of-turn reconcile, row by row: an equal row adopts its sequence and
 * keeps its object, a differing one takes the persisted content under its own
 * key, and a streamed row the DB lacks stays on screen.
 */
export function reconcileTurn(
  seg: TurnSegment,
  persisted: { raw: readonly PersistedRow[]; rows: readonly LogRow[] },
): TurnSegment {
  const rows = seg.frozen
    ? [...persisted.rows]
    : mergeRows(seg.log.rows, persisted.rows);
  const rowCreatedAt: Record<string, string> = {};
  rows.forEach((row, i) => {
    const at = createdAtOf(persisted.raw[i]);
    if (at) rowCreatedAt[row.key] = at;
  });
  const durations = persisted.raw
    .map((r) => (r as { duration_ms?: unknown }).duration_ms)
    .filter((d): d is number => typeof d === "number");
  return {
    ...seg,
    reconciled: true,
    ended: true,
    log: { ...closeBlocks(seg.log), rows },
    durationMs: durations.length ? Math.max(...durations) : seg.durationMs,
    createdAt:
      createdAtOf(persisted.raw[persisted.raw.length - 1]) ?? seg.createdAt,
    rowCreatedAt,
  };
}

/** A local prompt takes its persisted row's place: sequence, id and timestamp. */
export function withPromptRow(
  seg: UserSegment,
  row: PersistedRow,
): UserSegment {
  const rawId = (row as { id?: unknown }).id;
  return {
    ...seg,
    sequence: row.sequence ?? null,
    rawId: typeof rawId === "string" ? rawId : null,
    createdAt: createdAtOf(row),
  };
}

/** Rows persisted after the tail's end and before `start` that no segment holds. */
export function gapSegment(
  segments: readonly Segment[],
  start: number,
  rows: readonly PersistedRow[],
): RowsSegment | null {
  const tailEnd = tailEndSequence(segments);
  if (tailEnd === null) return null;
  const gap = rows.filter(
    (r) => (r.sequence ?? -1) > tailEnd && (r.sequence ?? -1) < start,
  );
  return gap.length
    ? { kind: "rows", key: `rows:${gap[0].sequence}`, rows: gap }
    : null;
}

export function tailEndSequence(segments: readonly Segment[]) {
  const last = segments[segments.length - 1];
  if (!last) return null;
  if (last.kind === "rows")
    return last.rows[last.rows.length - 1]?.sequence ?? null;
  if (last.kind === "user") return last.sequence;
  return last.log.rows[last.log.rows.length - 1]?.sequence ?? null;
}

/** Rows equal to the ones on screen keep their object and key; others take the persisted content under the on-screen key. */
export function mergeRows(
  live: readonly LogRow[],
  persisted: readonly LogRow[],
): LogRow[] {
  const differing = new Set(diffRows(live, persisted).map((d) => d.index));
  // A streamed row past the DB's end is one it lacks only if every row before
  // it matched; after a mismatch it is a persisted row at another index.
  const aligned = [...differing].every((i) => i >= persisted.length);
  const length = aligned ? live.length : 0;
  const merged: LogRow[] = [];
  for (let i = 0; i < Math.max(length, persisted.length); i++) {
    const a = live[i];
    const b = persisted[i];
    if (!b) merged.push(a);
    else if (!a) merged.push(b);
    else if (differing.has(i)) merged.push({ ...b, key: a.key });
    else merged.push(adoptPersisted(a, b));
  }
  return merged;
}

/** A resync's seed keeps the rows already on screen where they are equal. */
export function keepIdentity(
  live: readonly LogRow[],
  seeded: readonly LogRow[],
) {
  const differing = new Set(
    diffRows(live.slice(0, seeded.length), seeded).map((d) => d.index),
  );
  return seeded.map((row, i) => {
    const current = live[i];
    if (!current) return row;
    return differing.has(i)
      ? { ...row, key: current.key }
      : adoptPersisted(current, row);
  });
}

export function withStopMarker(log: TurnLog): TurnLog {
  const last = log.rows[log.rows.length - 1];
  const rows =
    last && isMarker(last)
      ? log.rows
      : [...log.rows, markerRow(`stop:${log.turnId}`, CANCELLED_MARKER)];
  return { ...closeBlocks(log), status: "finished", rows };
}

/** Rows that end in a marker row of their own, unless one is there already. */
export function withMarker(
  rows: readonly LogRow[],
  key: string,
  content: string,
) {
  const last = rows[rows.length - 1];
  return last && isMarker(last)
    ? [...rows]
    : [...rows, markerRow(key, content)];
}

export function closeBlocks(log: TurnLog): TurnLog {
  const open = Object.entries(log.blocks).filter(([, b]) => b.open);
  if (open.length === 0) return log;
  const blocks = { ...log.blocks };
  for (const [id, block] of open) blocks[id] = { ...block, open: false };
  return { ...log, blocks };
}

/** A block still writing or a tool call still waiting on its output. */
export function hasOpenParts(log: TurnLog) {
  return (
    Object.values(log.blocks).some((b) => b.open) ||
    Object.values(log.tools).some((t) => t.phase !== "output-available")
  );
}

export function persistedRows(
  messages: readonly unknown[] | null | undefined,
): PersistedRow[] {
  return (messages ?? [])
    .map((row) => row as PersistedRow)
    .filter((row) => typeof row.sequence === "number")
    .sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0));
}

/**
 * The prompt's row: the first user row in the tail whose text it carries
 * (the backend appends an attached-files block after it).
 */
export function locatePrompt(
  seg: UserSegment,
  rows: readonly PersistedRow[],
  from: number | null,
) {
  const text = seg.message.parts
    .map((p) => (p.type === "text" ? p.text : ""))
    .join("");
  const row = rows.find(
    (r) =>
      r.role === "user" &&
      (r.sequence ?? -1) >= (from ?? 0) &&
      typeof r.content === "string" &&
      r.content.startsWith(text),
  );
  return row ?? null;
}

export function lastCheckpoint(log: TurnLog): TurnCheckpoint | null {
  const last = log.checkpoints[log.checkpoints.length - 1];
  return last
    ? { entry_id: last.entryId, rows: last.rows, sequence: last.sequence }
    : null;
}

/** The checkpoint a 409 names, as the route writes it. */
export function checkpointOf(body: unknown): TurnCheckpoint | null {
  const checkpoint = (body as { checkpoint?: unknown } | null)?.checkpoint;
  if (!checkpoint || typeof checkpoint !== "object") return null;
  const { entry_id, rows, sequence } = checkpoint as Record<string, unknown>;
  return typeof entry_id === "string" &&
    typeof rows === "number" &&
    typeof sequence === "number"
    ? { entry_id, rows, sequence }
    : null;
}

function adoptPersisted(live: LogRow, persisted: LogRow): LogRow {
  const sameMetadata =
    JSON.stringify(live.metadata ?? null) ===
    JSON.stringify(persisted.metadata ?? live.metadata ?? null);
  if (live.sequence === persisted.sequence && sameMetadata) return live;
  return {
    ...live,
    sequence: persisted.sequence,
    metadata: persisted.metadata ?? live.metadata,
  };
}

function markerRow(key: string, content: string): LogRow {
  return {
    key,
    role: "assistant",
    content,
    toolCalls: [],
    toolCallId: null,
    sequence: null,
    metadata: null,
  };
}

function segmentStart(seg: Segment): number | null {
  if (seg.kind === "user") return seg.sequence;
  if (seg.kind === "rows") return seg.rows[0]?.sequence ?? null;
  return (
    seg.log.checkpoints[0]?.sequence ??
    seg.log.rows[0]?.sequence ??
    seg.startHint
  );
}

function createdAtOf(row: PersistedRow | undefined): string | null {
  const value = (row as { created_at?: unknown } | undefined)?.created_at;
  if (typeof value === "string") return value;
  return value instanceof Date ? value.toISOString() : null;
}
