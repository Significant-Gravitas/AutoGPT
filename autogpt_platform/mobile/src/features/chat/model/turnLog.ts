import type { TurnCheckpoint } from "../../../api/chat";
import type { Row, ToolCall } from "./rows";

/**
 * The client's model of one running turn: the rows the backend persists for
 * it, plus what rows cannot hold yet (open blocks, pending tools). The row
 * rules mirror the web client's `stream/turnLog.ts`, which in turn mirrors
 * the backend's `_dispatch_response` and pending-message persistence.
 */
export interface TurnLog {
  turnID: string | null;
  /** The last applied entry id; null until one is applied (a cursor of `0-0`). */
  cursor: string | null;
  status: "idle" | "running" | "finished" | "failed";
  messageID: string | null;
  rows: readonly Row[];
  blocks: Readonly<Record<string, BlockState>>;
  tools: Readonly<Record<string, ToolState>>;
  displayNames: Readonly<Record<string, string>>;
  assistantRow: number | null;
  hasToolResults: boolean;
  providerFailure: Record<string, unknown> | null;
  /** The latest `data-status` text, shown while nothing else is streaming. */
  statusText: string | null;
  protocolErrors: number;
}

export interface BlockState {
  kind: "text" | "reasoning";
  open: boolean;
  row: number | null;
}

export interface ToolState {
  name: string;
  phase: "input-streaming" | "input-available" | "output-available";
}

// Kept in step with backend `copilot/constants.py`.
export const COPILOT_ERROR_PREFIX = "[__COPILOT_ERROR_f7a1__]";
export const COPILOT_RETRYABLE_ERROR_PREFIX =
  "[__COPILOT_RETRYABLE_ERROR_a9c2__]";
export const COPILOT_SYSTEM_PREFIX = "[__COPILOT_SYSTEM_e3b0__]";
const RETRYABLE_ERROR_CODES = new Set([
  "transient_api_error",
  "empty_completion",
  "baseline_error",
]);
const CODE_PREFIX_RE = /^\s*\[code:([^\]]*)\] ?/;

export function emptyTurnLog(): TurnLog {
  return {
    turnID: null,
    cursor: null,
    status: "idle",
    messageID: null,
    rows: [],
    blocks: {},
    tools: {},
    displayNames: {},
    assistantRow: null,
    hasToolResults: false,
    providerFailure: null,
    statusText: null,
    protocolErrors: 0,
  };
}

/** Row rule 1: a text delta extends the assistant row, or opens one after tool results. */
export function appendText(
  log: TurnLog,
  blockID: string,
  delta: string,
): TurnLog {
  if (log.hasToolResults && log.assistantRow !== null) {
    return {
      ...pushRow(log, newRow(`assistant:${blockID}`, delta)),
      assistantRow: log.rows.length,
      hasToolResults: false,
    };
  }
  if (log.assistantRow === null) {
    return {
      ...pushRow(log, newRow(`assistant:${blockID}`, delta)),
      assistantRow: log.rows.length,
    };
  }
  const row = log.rows[log.assistantRow];
  return patchRow(log, log.assistantRow, { content: row.content + delta });
}

/** Row rule 2: every reasoning block is its own row, appended at its start. */
export function openReasoning(log: TurnLog, blockID: string): TurnLog {
  return pushRow(log, {
    ...newRow(`reasoning:${blockID}`, ""),
    role: "reasoning",
  });
}

/** Row rule 3: a reasoning delta extends its block's row. */
export function appendReasoning(
  log: TurnLog,
  row: number,
  delta: string,
): TurnLog {
  return patchRow(log, row, { content: log.rows[row].content + delta });
}

/** Row rule 4: a tool call joins the current assistant row, or opens one after tool results. */
export function addToolCall(
  log: TurnLog,
  call: Omit<ToolCall, "displayName">,
): TurnLog {
  const toolCall = { ...call, displayName: log.displayNames[call.id] ?? null };
  if (log.assistantRow === null || log.hasToolResults) {
    return {
      ...pushRow(log, {
        ...newRow(`assistant:${call.id}`, ""),
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
  toolCallID: string,
  output: unknown,
): TurnLog {
  if (log.rows.some((r) => r.role === "tool" && r.toolCallID === toolCallID))
    return log;
  const content = typeof output === "string" ? output : JSON.stringify(output);
  return {
    ...pushRow(log, {
      ...newRow(`tool:${toolCallID}`, content ?? ""),
      role: "tool",
      toolCallID,
    }),
    hasToolResults: true,
  };
}

/** Row rule 6: an error becomes a marker row, unless one already ends the turn. */
export function appendErrorMarker(
  log: TurnLog,
  entryID: string,
  errorText: string,
): TurnLog {
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
    ...newRow(`error:${entryID}`, `${prefix} ${display}`),
    metadata: failure ? { provider_failure: failure } : null,
  });
}

/** Row rule 7: a display name lands on its tool call, now or when the call arrives. */
export function setToolDisplayName(
  log: TurnLog,
  toolCallID: string,
  name: string,
): TurnLog {
  const displayNames = { ...log.displayNames, [toolCallID]: name };
  const index = log.rows.findIndex((r) =>
    r.toolCalls.some((c) => c.id === toolCallID),
  );
  if (index === -1) return { ...log, displayNames };
  const toolCalls = log.rows[index].toolCalls.map((c) =>
    c.id === toolCallID ? { ...c, displayName: name } : c,
  );
  return { ...patchRow(log, index, { toolCalls }), displayNames };
}

/** Row rule 8: drained follow-ups are user rows, keyed by pending id, where the hint arrived. */
export function appendDrainedUserRows(
  log: TurnLog,
  messages: readonly { id: string; content: string }[],
): TurnLog {
  const known = new Set(log.rows.map((r) => r.key));
  return messages
    .map((m) => ({
      ...newRow(`user:${m.id}`, m.content),
      role: "user" as const,
    }))
    .filter((row) => !known.has(row.key))
    .reduce(pushRow, log);
}

/**
 * The inverse of the row rules at a checkpoint: the turn's persisted rows
 * seed a log that the stream after the checkpoint continues. At a checkpoint
 * no block is open and no tool call is pending.
 */
export function seedTurnLog(args: {
  turnID: string;
  rows: readonly Row[];
  checkpoint: TurnCheckpoint | null;
}): TurnLog {
  const base: TurnLog = {
    ...emptyTurnLog(),
    turnID: args.turnID,
    status: "running",
  };
  const checkpoint = args.checkpoint;
  if (!checkpoint) return base;
  const seeded = args.rows
    .filter(
      (r) =>
        r.sequence !== null &&
        r.sequence >= checkpoint.sequence &&
        r.sequence < checkpoint.sequence + checkpoint.rows,
    )
    .sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0));
  const lastAssistant = seeded.findLastIndex(
    (r) => r.role === "assistant" && !isMarker(r),
  );
  const answered = new Set(seeded.map((r) => r.toolCallID));
  const tools: Record<string, ToolState> = {};
  for (const call of seeded.flatMap((r) => r.toolCalls)) {
    tools[call.id] = {
      name: call.name,
      phase: answered.has(call.id) ? "output-available" : "input-available",
    };
  }
  return {
    ...base,
    cursor: checkpoint.entryID,
    rows: seeded,
    tools,
    assistantRow: lastAssistant === -1 ? null : lastAssistant,
    hasToolResults:
      lastAssistant !== -1 &&
      seeded.slice(lastAssistant + 1).some((r) => r.role === "tool"),
  };
}

export function isMarker(row: Row): boolean {
  return (
    row.role === "assistant" &&
    [
      COPILOT_ERROR_PREFIX,
      COPILOT_RETRYABLE_ERROR_PREFIX,
      COPILOT_SYSTEM_PREFIX,
    ].some((prefix) => row.content.startsWith(prefix))
  );
}

function newRow(key: string, content: string): Row {
  return {
    key,
    role: "assistant",
    content,
    toolCalls: [],
    toolCallID: null,
    sequence: null,
    metadata: null,
    createdAt: null,
  };
}

function pushRow(log: TurnLog, row: Row): TurnLog {
  return { ...log, rows: [...log.rows, row] };
}

function patchRow(log: TurnLog, index: number, patch: Partial<Row>): TurnLog {
  const rows = log.rows.slice();
  rows[index] = { ...rows[index], ...patch };
  return { ...log, rows };
}
