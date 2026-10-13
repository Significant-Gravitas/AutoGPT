/**
 * The client's model of one running turn: the rows the backend persists for
 * it, plus what rows cannot hold (open blocks, pending tools, transient parts).
 *
 * The row rules mirror `_dispatch_response` in `backend/copilot/sdk/service.py`
 * and `persist_pending_as_user_rows`; a change there that is not mirrored here
 * turns the replay-equivalence test red.
 */

export type RowRole = "assistant" | "reasoning" | "tool" | "user";

export interface LogToolCall {
  id: string;
  name: string;
  input: unknown;
  displayName: string | null;
}

export interface LogRow {
  /** Stable for the life of the row: a block, tool call or pending message id. */
  key: string;
  role: RowRole;
  content: string;
  toolCalls: readonly LogToolCall[];
  toolCallId: string | null;
  sequence: number | null;
  metadata: Record<string, unknown> | null;
}

export interface BlockState {
  kind: "text" | "reasoning";
  open: boolean;
  /** The row this block writes into; a text block has none until its first delta. */
  row: number | null;
}

export interface ToolState {
  name: string;
  phase: "input-streaming" | "input-available" | "output-available";
}

export interface OverlayPart {
  entryId: string;
  type: string;
  data: unknown;
  /** How many rows existed when it arrived; it renders after the last of them. */
  anchor: number;
}

export interface Checkpoint {
  entryId: string;
  rows: number;
  sequence: number;
  digest: string;
  /** The digest input over this log's rows at the moment the checkpoint applied. */
  canonical: string;
}

export interface ProtocolError {
  entryId: string;
  chunkType: string;
  reason: string;
}

export interface TurnLog {
  turnId: string | null;
  /** The last applied entry id; null until one is applied (a cursor of `0-0`). */
  cursor: string | null;
  status: "idle" | "running" | "finished" | "failed";
  messageId: string | null;
  rows: readonly LogRow[];
  blocks: Readonly<Record<string, BlockState>>;
  tools: Readonly<Record<string, ToolState>>;
  displayNames: Readonly<Record<string, string>>;
  /** The engine's current assistant row, once appended. */
  assistantRow: number | null;
  /** A tool result landed since the current assistant row opened. */
  hasToolResults: boolean;
  overlay: readonly OverlayPart[];
  /** The rows and overlay as the open step found them; null when no step is open. */
  openStep: { rows: readonly LogRow[]; overlay: readonly OverlayPart[] } | null;
  providerFailure: Record<string, unknown> | null;
  checkpoints: readonly Checkpoint[];
  protocolErrors: readonly ProtocolError[];
}

// Kept in step with `backend/copilot/constants.py`.
export const COPILOT_ERROR_PREFIX = "[__COPILOT_ERROR_f7a1__]";
export const COPILOT_RETRYABLE_ERROR_PREFIX =
  "[__COPILOT_RETRYABLE_ERROR_a9c2__]";
const COPILOT_SYSTEM_PREFIX = "[__COPILOT_SYSTEM_e3b0__]";
// SDK `_RETRYABLE_STREAM_ERROR_CODES`, plus the baseline's default-retryable code.
const RETRYABLE_ERROR_CODES = new Set([
  "transient_api_error",
  "empty_completion",
  "baseline_error",
]);
const CODE_PREFIX_RE = /^\s*\[code:([^\]]*)\] ?/;

export function emptyTurnLog(): TurnLog {
  return {
    turnId: null,
    cursor: null,
    status: "idle",
    messageId: null,
    rows: [],
    blocks: {},
    tools: {},
    displayNames: {},
    assistantRow: null,
    hasToolResults: false,
    overlay: [],
    openStep: null,
    providerFailure: null,
    checkpoints: [],
    protocolErrors: [],
  };
}

/** Row rule 1: a text delta extends the assistant row, or opens one after tool results. */
export function appendText(log: TurnLog, blockId: string, delta: string) {
  if (log.hasToolResults && log.assistantRow !== null) {
    return {
      ...pushRow(log, assistantRow(`assistant:${blockId}`, delta)),
      assistantRow: log.rows.length,
      hasToolResults: false,
    };
  }
  if (log.assistantRow === null) {
    return {
      ...pushRow(log, assistantRow(`assistant:${blockId}`, delta)),
      assistantRow: log.rows.length,
    };
  }
  const row = log.rows[log.assistantRow];
  return patchRow(log, log.assistantRow, { content: row.content + delta });
}

/** Row rule 2: every reasoning block is its own row, appended at its start. */
export function openReasoning(log: TurnLog, blockId: string) {
  return pushRow(log, {
    ...assistantRow(`reasoning:${blockId}`, ""),
    role: "reasoning",
  });
}

/** Row rule 3: a reasoning delta extends its block's row. */
export function appendReasoning(log: TurnLog, row: number, delta: string) {
  return patchRow(log, row, { content: log.rows[row].content + delta });
}

/** Row rule 4: a tool call joins the current assistant row, or opens one after tool results. */
export function addToolCall(
  log: TurnLog,
  call: Omit<LogToolCall, "displayName">,
) {
  const toolCall = { ...call, displayName: log.displayNames[call.id] ?? null };
  if (log.assistantRow === null || log.hasToolResults) {
    return {
      ...pushRow(log, {
        ...assistantRow(`assistant:${call.id}`, ""),
        toolCalls: [toolCall],
      }),
      assistantRow: log.rows.length,
      hasToolResults: false,
    };
  }
  const row = log.rows[log.assistantRow];
  return patchRow(log, log.assistantRow, {
    toolCalls: [...row.toolCalls, toolCall],
  });
}

/** Row rule 5: a tool result is its own row, once per call. */
export function addToolResult(
  log: TurnLog,
  toolCallId: string,
  output: unknown,
) {
  if (log.rows.some((r) => r.role === "tool" && r.toolCallId === toolCallId)) {
    return log;
  }
  return {
    ...pushRow(log, {
      ...assistantRow(`tool:${toolCallId}`, toolContent(output)),
      role: "tool",
      toolCallId,
    }),
    hasToolResults: true,
  };
}

/** Row rule 6: an error becomes a marker row, unless one already ends the turn. */
export function appendErrorMarker(
  log: TurnLog,
  entryId: string,
  errorText: string,
) {
  const failure = log.providerFailure;
  const cleared = { ...log, providerFailure: null, status: "failed" as const };
  const last = log.rows[log.rows.length - 1];
  if (last && isMarker(last)) return cleared;
  const code = CODE_PREFIX_RE.exec(errorText)?.[1] ?? null;
  const display =
    typeof failure?.message === "string"
      ? failure.message
      : errorText.replace(CODE_PREFIX_RE, "");
  const retryable = failure
    ? failure.retryable === true
    : code !== null && RETRYABLE_ERROR_CODES.has(code);
  const prefix = retryable
    ? COPILOT_RETRYABLE_ERROR_PREFIX
    : COPILOT_ERROR_PREFIX;
  return pushRow(cleared, {
    ...assistantRow(`error:${entryId}`, `${prefix} ${display}`),
    metadata: failure ? { provider_failure: failure } : null,
  });
}

/** Row rule 7: a display name lands on its tool call, now or when the call arrives. */
export function setToolDisplayName(
  log: TurnLog,
  toolCallId: string,
  name: string,
) {
  const displayNames = { ...log.displayNames, [toolCallId]: name };
  const index = log.rows.findIndex((r) =>
    r.toolCalls.some((c) => c.id === toolCallId),
  );
  if (index === -1) return { ...log, displayNames };
  const toolCalls = log.rows[index].toolCalls.map((c) =>
    c.id === toolCallId ? { ...c, displayName: name } : c,
  );
  return { ...patchRow(log, index, { toolCalls }), displayNames };
}

/**
 * Row rule 8: drained follow-ups are user rows, keyed by pending message id,
 * at the hint's place in the stream.
 */
export function appendDrainedUserRows(
  log: TurnLog,
  messages: readonly { id: string; content: string }[],
) {
  const known = new Set(log.rows.map((r) => r.key));
  const rows = messages
    .map((m) => ({
      ...assistantRow(`user:${m.id}`, m.content),
      role: "user" as const,
    }))
    .filter((row) => !known.has(row.key));
  return rows.reduce(pushRow, log);
}

/**
 * The inverse of the row rules at a checkpoint: the persisted rows of a turn
 * seed a log that the stream tail after the checkpoint continues. At a
 * checkpoint no block is open and no tool call is pending.
 */
export function seedTurnLog({
  turnId,
  rows,
  checkpoint,
}: {
  turnId: string;
  rows: readonly PersistedRow[];
  checkpoint: { entry_id: string; rows: number; sequence: number } | null;
}): TurnLog {
  const base = { ...emptyTurnLog(), turnId, status: "running" as const };
  if (!checkpoint) return base;
  const { sequence, rows: count } = checkpoint;
  const seeded = rows
    .filter(
      (r) =>
        typeof r.sequence === "number" &&
        r.sequence >= sequence &&
        r.sequence < sequence + count,
    )
    .sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0))
    .map(logRowFromPersisted);
  const lastAssistant = seeded.findLastIndex(
    (r) => r.role === "assistant" && !isMarker(r),
  );
  const tools: Record<string, ToolState> = {};
  const answered = new Set(seeded.map((r) => r.toolCallId));
  for (const call of seeded.flatMap((r) => r.toolCalls)) {
    tools[call.id] = {
      name: call.name,
      phase: answered.has(call.id) ? "output-available" : "input-available",
    };
  }
  return {
    ...base,
    cursor: checkpoint.entry_id,
    rows: seeded,
    tools,
    assistantRow: lastAssistant === -1 ? null : lastAssistant,
    hasToolResults:
      lastAssistant !== -1 &&
      seeded.slice(lastAssistant + 1).some((r) => r.role === "tool"),
  };
}

/** The checkpoint digest's input, as `stream_checkpoint.rows_digest` builds it. */
export function canonicalRows(rows: readonly LogRow[]): string {
  return JSON.stringify(
    rows.map((row) => [
      row.role,
      (row.role === "tool" ? row.toolCallId : row.content) || "",
      row.toolCalls.map((call) => [call.id, call.name]),
    ]),
  );
}

export function isMarker(row: LogRow): boolean {
  return (
    row.role === "assistant" &&
    [
      COPILOT_ERROR_PREFIX,
      COPILOT_RETRYABLE_ERROR_PREFIX,
      COPILOT_SYSTEM_PREFIX,
    ].some((prefix) => row.content.startsWith(prefix))
  );
}

/** A row as `GET /sessions/{id}` returns it. */
export interface PersistedRow {
  role: string;
  content?: string | null;
  tool_call_id?: string | null;
  tool_calls?: unknown[] | null;
  sequence?: number | null;
  metadata?: Record<string, unknown> | null;
}

export function logRowFromPersisted(row: PersistedRow): LogRow {
  const role = (["assistant", "reasoning", "tool", "user"] as const).find(
    (r) => r === row.role,
  );
  return {
    key: `seq:${row.sequence}`,
    role: role ?? "assistant",
    content: row.content ?? "",
    toolCalls: (row.tool_calls ?? []).map(persistedToolCall),
    toolCallId: row.tool_call_id ?? null,
    sequence: row.sequence ?? null,
    metadata: row.metadata ?? null,
  };
}

export function parseJsonLoose(value: string): unknown {
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}

function persistedToolCall(raw: unknown): LogToolCall {
  const call = (raw ?? {}) as {
    id?: unknown;
    display_name?: unknown;
    function?: { name?: unknown; arguments?: unknown };
  };
  const args = call.function?.arguments;
  return {
    id: String(call.id ?? ""),
    name: String(call.function?.name ?? ""),
    input: typeof args === "string" ? parseJsonLoose(args) : (args ?? {}),
    displayName:
      typeof call.display_name === "string" ? call.display_name : null,
  };
}

function assistantRow(key: string, content: string): LogRow {
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

// The backend stores a dict output as `json.dumps`, whose spacing differs;
// only the digest-free finish comparison reads it, and it parses both sides.
function toolContent(output: unknown): string {
  return typeof output === "string" ? output : JSON.stringify(output);
}

function pushRow(log: TurnLog, row: LogRow): TurnLog {
  return { ...log, rows: [...log.rows, row] };
}

function patchRow(
  log: TurnLog,
  index: number,
  patch: Partial<LogRow>,
): TurnLog {
  const rows = log.rows.slice();
  rows[index] = { ...rows[index], ...patch };
  return { ...log, rows };
}
