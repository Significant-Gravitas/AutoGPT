import type { WireChunk } from "../../../api/sse";
import { asRecord, asString } from "../../../lib/json";
import {
  addToolCall,
  addToolResult,
  appendDrainedUserRows,
  appendErrorMarker,
  appendReasoning,
  appendText,
  openReasoning,
  setToolDisplayName,
  type TurnLog,
} from "./turnLog";

export interface StreamEntry {
  turn: string;
  entryID: string;
  chunk: WireChunk;
}

/**
 * Applies one stored entry. Pure and idempotent over any suffix: an entry at
 * or before the cursor is dropped before any rule runs, so overlapping
 * deliveries after a reconnect converge. Unknown chunk types are counted and
 * ignored; they never stop the turn from rendering.
 */
export function applyEntry(log: TurnLog, entry: StreamEntry): TurnLog {
  if (log.turnID !== null && entry.turn !== log.turnID)
    return protocolError(log);
  if (log.cursor !== null && compareEntryIDs(entry.entryID, log.cursor) <= 0)
    return log;
  const advanced: TurnLog = {
    ...log,
    turnID: entry.turn,
    cursor: entry.entryID,
    status: log.status === "idle" ? "running" : log.status,
  };
  return applyChunk(advanced, entry.entryID, entry.chunk);
}

/** Redis stream ids are `<ms>-<seq>`, ordered numerically on both halves. */
export function compareEntryIDs(a: string, b: string): number {
  const [aMs, aSeq] = a.split("-").map(Number);
  const [bMs, bSeq] = b.split("-").map(Number);
  return aMs !== bMs ? aMs - bMs : (aSeq ?? 0) - (bSeq ?? 0);
}

function applyChunk(log: TurnLog, entryID: string, chunk: WireChunk): TurnLog {
  switch (chunk.type) {
    case "start": {
      const messageID = str(chunk.messageId);
      // An auto-continue streams a second message into the same turn.
      if (!messageID || messageID === log.messageID) return log;
      return { ...log, messageID, assistantRow: null, hasToolResults: false };
    }
    case "finish":
      return {
        ...log,
        status: log.status === "failed" ? "failed" : "finished",
      };
    case "start-step":
      return log;
    case "finish-step":
      return closeOpenBlocks(log);
    case "text-start":
    case "reasoning-start":
      return startBlock(log, chunk);
    case "text-delta":
    case "reasoning-delta":
      return extendBlock(log, chunk);
    case "text-end":
    case "reasoning-end": {
      const block = log.blocks[str(chunk.id)];
      if (!block || !block.open) return log;
      return {
        ...log,
        blocks: { ...log.blocks, [str(chunk.id)]: { ...block, open: false } },
      };
    }
    case "tool-input-start":
      return startTool(log, chunk);
    case "tool-input-available":
      return receiveToolInput(log, chunk);
    case "tool-output-available":
      return receiveToolOutput(log, chunk);
    case "error":
      return appendErrorMarker(log, entryID, str(chunk.errorText));
    case "data-tool-display": {
      const data = asRecord(chunk.data);
      return setToolDisplayName(
        log,
        str(data?.toolCallId),
        str(data?.displayName),
      );
    }
    case "data-pending-drained":
      return appendDrainedUserRows(log, drainedMessages(chunk.data));
    case "data-status": {
      const message = asString(asRecord(chunk.data)?.message);
      return message ? { ...log, statusText: message } : log;
    }
    case "data-provider-failure":
      return { ...log, providerFailure: asRecord(chunk.data) };
    default:
      // Other data parts (checkpoints, compaction progress, mode changes,
      // dream operations) are transient and carry nothing this view draws.
      return chunk.type.startsWith("data-") ? log : protocolError(log);
  }
}

function closeOpenBlocks(log: TurnLog): TurnLog {
  const open = Object.entries(log.blocks).filter(([, block]) => block.open);
  if (open.length === 0) return log;
  const blocks = { ...log.blocks };
  for (const [id, block] of open) blocks[id] = { ...block, open: false };
  return { ...log, blocks };
}

function startBlock(log: TurnLog, chunk: WireChunk): TurnLog {
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

function extendBlock(log: TurnLog, chunk: WireChunk): TurnLog {
  const id = str(chunk.id);
  const block = log.blocks[id];
  const kind = chunk.type === "text-delta" ? "text" : "reasoning";
  if (!block || block.kind !== kind || !block.open) return protocolError(log);
  const delta = str(chunk.delta);
  if (kind === "reasoning" && block.row !== null)
    return appendReasoning(log, block.row, delta);
  const written = appendText(log, id, delta);
  return {
    ...written,
    blocks: { ...log.blocks, [id]: { ...block, row: written.assistantRow } },
  };
}

function startTool(log: TurnLog, chunk: WireChunk): TurnLog {
  const id = str(chunk.toolCallId);
  if (!id || log.tools[id]) return log;
  return {
    ...log,
    tools: {
      ...log.tools,
      [id]: { name: str(chunk.toolName), phase: "input-streaming" },
    },
  };
}

function receiveToolInput(log: TurnLog, chunk: WireChunk): TurnLog {
  const id = str(chunk.toolCallId);
  if (!id) return protocolError(log);
  const known = log.tools[id];
  if (known && known.phase !== "input-streaming") return log;
  const name = str(chunk.toolName) || known?.name || "";
  return {
    ...addToolCall(log, { id, name, input: chunk.input ?? {} }),
    tools: { ...log.tools, [id]: { name, phase: "input-available" } },
  };
}

// An output for a call this log never saw is still a row the backend persists.
function receiveToolOutput(log: TurnLog, chunk: WireChunk): TurnLog {
  const id = str(chunk.toolCallId);
  if (!id) return protocolError(log);
  const known = log.tools[id];
  if (known?.phase === "output-available") return log;
  return {
    ...addToolResult(log, id, chunk.output),
    tools: {
      ...log.tools,
      [id]: { name: known?.name ?? "", phase: "output-available" },
    },
  };
}

function drainedMessages(data: unknown) {
  const messages = asRecord(data)?.messages;
  return (Array.isArray(messages) ? messages : []).map((m) => ({
    id: str(asRecord(m)?.id),
    content: str(asRecord(m)?.content),
  }));
}

function protocolError(log: TurnLog): TurnLog {
  return { ...log, protocolErrors: log.protocolErrors + 1 };
}

function str(value: unknown): string {
  return typeof value === "string" ? value : "";
}
