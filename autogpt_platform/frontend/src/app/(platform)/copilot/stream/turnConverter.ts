import {
  addToolCall,
  addToolResult,
  appendDrainedUserRows,
  appendErrorMarker,
  appendReasoning,
  appendText,
  canonicalRows,
  openReasoning,
  setToolDisplayName,
  type TurnLog,
} from "./turnLog";

/** One JSON chunk as it arrives on the wire. */
export interface WireChunk {
  type: string;
  [field: string]: unknown;
}

/** An entry-bearing frame: its SSE id is `<turn>:<entry>`. */
export interface StreamEntry {
  turn: string;
  entryId: string;
  chunk: WireChunk;
}

/**
 * Apply one stored entry. Pure, keyed by part identity, and idempotent over
 * any suffix: an entry at or before the cursor is dropped before any rule
 * runs, so overlapping deliveries converge. A protocol error is recorded and
 * never stops the fold.
 */
export function applyEntry(log: TurnLog, entry: StreamEntry): TurnLog {
  if (log.turnId !== null && entry.turn !== log.turnId) {
    return protocolError(log, entry.entryId, entry.chunk.type, "another turn");
  }
  if (log.cursor !== null && compareEntryIds(entry.entryId, log.cursor) <= 0) {
    return log;
  }
  const advanced: TurnLog = {
    ...log,
    turnId: entry.turn,
    cursor: entry.entryId,
    status: log.status === "idle" ? "running" : log.status,
  };
  return applyChunk(advanced, entry.entryId, entry.chunk);
}

export function foldEntries(log: TurnLog, entries: Iterable<StreamEntry>) {
  let next = log;
  for (const entry of entries) next = applyEntry(next, entry);
  return next;
}

/** Redis stream ids are `<ms>-<seq>`, ordered numerically on both halves. */
export function compareEntryIds(a: string, b: string): number {
  const [aMs, aSeq] = a.split("-").map(Number);
  const [bMs, bSeq] = b.split("-").map(Number);
  return aMs !== bMs ? aMs - bMs : (aSeq ?? 0) - (bSeq ?? 0);
}

function applyChunk(log: TurnLog, entryId: string, chunk: WireChunk): TurnLog {
  switch (chunk.type) {
    case "start":
      return startMessage(log, str(chunk.messageId));
    case "finish":
      return {
        ...log,
        openStep: null,
        status: log.status === "failed" ? "failed" : "finished",
      };
    case "start-step":
      return { ...log, openStep: { rows: log.rows, overlay: log.overlay } };
    case "finish-step":
      return finishStep(log, entryId);
    case "text-start":
    case "reasoning-start":
      return startBlock(log, entryId, chunk);
    case "text-delta":
    case "reasoning-delta":
      return extendBlock(log, entryId, chunk);
    case "text-end":
    case "reasoning-end":
      return endBlock(log, entryId, chunk);
    case "tool-input-start":
      return startTool(log, chunk);
    case "tool-input-available":
      return receiveToolInput(log, entryId, chunk);
    case "tool-output-available":
      return receiveToolOutput(log, entryId, chunk);
    case "error":
      return appendErrorMarker(log, entryId, str(chunk.errorText));
    case "data-tool-display": {
      const data = record(chunk.data);
      return setToolDisplayName(
        log,
        str(data.toolCallId),
        str(data.displayName),
      );
    }
    case "data-pending-drained":
      return appendDrainedUserRows(log, drainedMessages(chunk.data));
    case "data-checkpoint":
      return recordCheckpoint(log, entryId, record(chunk.data));
    case "data-cursor":
      return log;
    case "data-provider-failure":
      return {
        ...addOverlay(log, entryId, chunk),
        providerFailure: record(chunk.data),
      };
    default:
      if (chunk.type.startsWith("data-"))
        return addOverlay(log, entryId, chunk);
      return protocolError(log, entryId, chunk.type, "unknown chunk type");
  }
}

// An auto-continue streams a second message into the same turn; the engine
// gives each message a fresh assistant row, as it does each call.
function startMessage(log: TurnLog, messageId: string): TurnLog {
  if (!messageId || messageId === log.messageId) return log;
  return { ...log, messageId, assistantRow: null, hasToolResults: false };
}

// The backend ends every open block before a step closes (#14762).
function finishStep(log: TurnLog, entryId: string): TurnLog {
  const open = Object.entries(log.blocks).filter(([, b]) => b.open);
  const closed = { ...log, openStep: null };
  if (open.length === 0) return closed;
  const blocks = { ...log.blocks };
  for (const [id, block] of open) blocks[id] = { ...block, open: false };
  return protocolError(
    { ...closed, blocks },
    entryId,
    "finish-step",
    "block open",
  );
}

function startBlock(log: TurnLog, entryId: string, chunk: WireChunk): TurnLog {
  const id = str(chunk.id);
  if (log.blocks[id]) return log;
  if (chunk.type === "text-start") {
    return {
      ...log,
      blocks: { ...log.blocks, [id]: { kind: "text", open: true, row: null } },
    };
  }
  const opened = openReasoning(log, id);
  return {
    ...opened,
    blocks: {
      ...log.blocks,
      [id]: { kind: "reasoning", open: true, row: log.rows.length },
    },
  };
}

function extendBlock(log: TurnLog, entryId: string, chunk: WireChunk): TurnLog {
  const id = str(chunk.id);
  const block = log.blocks[id];
  const kind = chunk.type === "text-delta" ? "text" : "reasoning";
  if (!block || block.kind !== kind) {
    return protocolError(log, entryId, chunk.type, "unknown block");
  }
  if (!block.open)
    return protocolError(log, entryId, chunk.type, "closed block");
  const delta = str(chunk.delta);
  if (kind === "reasoning" && block.row !== null) {
    return appendReasoning(log, block.row, delta);
  }
  const written = appendText(log, id, delta);
  return {
    ...written,
    blocks: { ...log.blocks, [id]: { ...block, row: written.assistantRow } },
  };
}

function endBlock(log: TurnLog, entryId: string, chunk: WireChunk): TurnLog {
  const id = str(chunk.id);
  const block = log.blocks[id];
  if (!block) return protocolError(log, entryId, chunk.type, "unknown block");
  if (!block.open) return log;
  return { ...log, blocks: { ...log.blocks, [id]: { ...block, open: false } } };
}

function startTool(log: TurnLog, chunk: WireChunk): TurnLog {
  const id = str(chunk.toolCallId);
  if (log.tools[id]) return log;
  return {
    ...log,
    tools: {
      ...log.tools,
      [id]: { name: str(chunk.toolName), phase: "input-streaming" },
    },
  };
}

function receiveToolInput(
  log: TurnLog,
  entryId: string,
  chunk: WireChunk,
): TurnLog {
  const id = str(chunk.toolCallId);
  const input = chunk.input ?? {};
  const known = log.tools[id];
  if (known && known.phase !== "input-streaming") {
    const call = log.rows.flatMap((r) => r.toolCalls).find((c) => c.id === id);
    return call && !jsonEqual(call.input, input)
      ? protocolError(log, entryId, chunk.type, "input changed")
      : log;
  }
  const name = str(chunk.toolName) || known?.name || "";
  return {
    ...addToolCall(log, { id, name, input }),
    tools: { ...log.tools, [id]: { name, phase: "input-available" } },
  };
}

// An output for a call the log never saw is still a row the backend persists,
// so it is kept as well as reported.
function receiveToolOutput(
  log: TurnLog,
  entryId: string,
  chunk: WireChunk,
): TurnLog {
  const id = str(chunk.toolCallId);
  const known = log.tools[id];
  if (known?.phase === "output-available") return log;
  const withRow = {
    ...addToolResult(log, id, chunk.output),
    tools: {
      ...log.tools,
      [id]: { name: known?.name ?? "", phase: "output-available" as const },
    },
  };
  return known?.phase === "input-available"
    ? withRow
    : protocolError(withRow, entryId, chunk.type, "unknown tool call");
}

function recordCheckpoint(
  log: TurnLog,
  entryId: string,
  data: Record<string, unknown>,
): TurnLog {
  const count = num(data.rows);
  const sequence = num(data.sequence);
  const rows = log.rows.map((row, i) =>
    i < count && row.sequence !== sequence + i
      ? { ...row, sequence: sequence + i }
      : row,
  );
  const checkpoint = {
    entryId,
    rows: count,
    sequence,
    digest: str(data.digest),
    canonical: canonicalRows(rows.slice(0, count)),
  };
  return { ...log, rows, checkpoints: [...log.checkpoints, checkpoint] };
}

function addOverlay(log: TurnLog, entryId: string, chunk: WireChunk): TurnLog {
  const part = {
    entryId,
    type: chunk.type,
    data: chunk.data,
    anchor: log.rows.length,
  };
  return { ...log, overlay: [...log.overlay, part] };
}

function protocolError(
  log: TurnLog,
  entryId: string,
  chunkType: string,
  reason: string,
): TurnLog {
  return {
    ...log,
    protocolErrors: [...log.protocolErrors, { entryId, chunkType, reason }],
  };
}

function drainedMessages(data: unknown) {
  const messages = record(data).messages;
  return (Array.isArray(messages) ? messages : []).map((m) => ({
    id: str(record(m).id),
    content: str(record(m).content),
  }));
}

export function jsonEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== "object" || typeof b !== "object" || !a || !b) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const aKeys = Object.keys(a);
  const bRecord = b as Record<string, unknown>;
  return (
    aKeys.length === Object.keys(b).length &&
    aKeys.every((k) => jsonEqual((a as Record<string, unknown>)[k], bRecord[k]))
  );
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function str(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function num(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}
