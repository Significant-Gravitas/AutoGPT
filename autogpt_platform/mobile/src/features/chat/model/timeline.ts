import { parseJSONLoose } from "../../../lib/json";
import type { ToolPart } from "../tools/presentation";
import { splitAttachments, type MessageAttachment } from "./attachments";
import type { Row } from "./rows";
import {
  COPILOT_ERROR_PREFIX,
  COPILOT_RETRYABLE_ERROR_PREFIX,
  COPILOT_SYSTEM_PREFIX,
  type TurnLog,
} from "./turnLog";

/** A message typed on this device that history doesn't hold yet. */
export interface LocalMessage {
  /** The idempotency key sent with it; a manual retry reuses it. */
  messageID: string;
  text: string;
  state: "sending" | "accepted" | "failed";
  error: string | null;
  createdAt: number;
}

export type AssistantPart =
  | { kind: "text"; key: string; text: string; streaming: boolean }
  | { kind: "reasoning"; key: string; text: string; streaming: boolean }
  | { kind: "tool"; key: string; tool: ToolPart }
  | { kind: "error"; key: string; text: string; retryable: boolean }
  | { kind: "system"; key: string; text: string };

export type TimelineItem =
  | {
      kind: "user";
      key: string;
      text: string;
      attachments: MessageAttachment[];
      local: LocalMessage | null;
    }
  | { kind: "assistant"; key: string; parts: AssistantPart[]; live: boolean }
  | { kind: "work"; key: string; text: string };

type Unanswered = "running" | "unrecorded" | "interrupted";

/** Persisted rows; built once per history change, not per streamed delta. */
export function historyItems(
  history: readonly Row[],
  turnRunning: boolean,
): TimelineItem[] {
  return rowsToItems(history, {
    keyPrefix: "",
    live: false,
    outputs: toolOutputs(history),
    unanswered: turnRunning ? "running" : "unrecorded",
    streamingRows: new Set(),
    skipUserContents: [],
  });
}

/** The running (or just finished) turn, rebuilt as its stream advances. */
export function liveItems(
  live: TurnLog,
  replayedUserContents: readonly string[],
): TimelineItem[] {
  const streamingRows = new Set<number>();
  for (const block of Object.values(live.blocks)) {
    if (block.open && block.row !== null) streamingRows.add(block.row);
  }
  return rowsToItems(live.rows, {
    keyPrefix: "live:",
    live: true,
    outputs: toolOutputs(live.rows),
    unanswered:
      live.status === "running" || live.status === "idle"
        ? "running"
        : "interrupted",
    streamingRows,
    skipUserContents: [...replayedUserContents],
  });
}

export function localItem(local: LocalMessage): TimelineItem {
  return {
    kind: "user",
    key: `local:${local.messageID}`,
    text: local.text,
    attachments: [],
    local,
  };
}

interface ItemOptions {
  keyPrefix: string;
  live: boolean;
  outputs: Map<string, unknown>;
  unanswered: Unanswered;
  streamingRows: Set<number>;
  skipUserContents: string[];
}

function toolOutputs(rows: readonly Row[]) {
  const outputs = new Map<string, unknown>();
  for (const row of rows) {
    if (row.role === "tool" && row.toolCallID)
      outputs.set(row.toolCallID, parseJSONLoose(row.content));
  }
  return outputs;
}

function rowsToItems(
  rows: readonly Row[],
  options: ItemOptions,
): TimelineItem[] {
  const items: TimelineItem[] = [];
  let group: {
    kind: "assistant";
    key: string;
    parts: AssistantPart[];
    live: boolean;
  } | null = null;
  rows.forEach((row, index) => {
    if (row.role === "tool") return;
    if (row.role === "user") {
      group = null;
      if (row.metadata?.hidden === true) return;
      const skip = options.skipUserContents.indexOf(row.content);
      if (skip !== -1) {
        options.skipUserContents.splice(skip, 1);
        return;
      }
      const { text, attachments } = splitAttachments(row.content);
      items.push({
        kind: "user",
        key: options.keyPrefix + row.key,
        text,
        attachments,
        local: null,
      });
      return;
    }
    if (row.metadata?.kind === "expert_run") {
      group = null;
      if (row.content.trim()) {
        items.push({
          kind: "work",
          key: options.keyPrefix + row.key,
          text: row.content.trim(),
        });
      }
      return;
    }
    const parts = rowParts(row, options.streamingRows.has(index), options);
    if (parts.length === 0) return;
    if (!group) {
      group = {
        kind: "assistant",
        key: options.keyPrefix + row.key,
        parts: [],
        live: options.live,
      };
      items.push(group);
    }
    group.parts.push(...parts);
  });
  return items;
}

function rowParts(
  row: Row,
  streaming: boolean,
  options: ItemOptions,
): AssistantPart[] {
  const parts: AssistantPart[] = [];
  const content = row.content;
  if (row.role === "reasoning") {
    if (content.trim())
      parts.push({ kind: "reasoning", key: row.key, text: content, streaming });
  } else if (content.startsWith(COPILOT_RETRYABLE_ERROR_PREFIX)) {
    parts.push({
      kind: "error",
      key: row.key,
      text: markerText(content, COPILOT_RETRYABLE_ERROR_PREFIX),
      retryable: true,
    });
  } else if (content.startsWith(COPILOT_ERROR_PREFIX)) {
    parts.push({
      kind: "error",
      key: row.key,
      text: markerText(content, COPILOT_ERROR_PREFIX),
      retryable: false,
    });
  } else if (content.startsWith(COPILOT_SYSTEM_PREFIX)) {
    parts.push({
      kind: "system",
      key: row.key,
      text: markerText(content, COPILOT_SYSTEM_PREFIX),
    });
  } else if (content.trim()) {
    parts.push({ kind: "text", key: row.key, text: content, streaming });
  }
  for (const call of row.toolCalls) {
    if (!call.id) continue;
    const output = options.outputs.get(call.id);
    parts.push({
      kind: "tool",
      key: `${row.key}:${call.id}`,
      tool: {
        toolCallID: call.id,
        toolName: call.name,
        displayName: call.displayName,
        input: call.input,
        output,
        state: output !== undefined ? "done" : options.unanswered,
      },
    });
  }
  return parts;
}

function markerText(content: string, prefix: string): string {
  return content.slice(prefix.length).trim() || "Something went wrong.";
}
