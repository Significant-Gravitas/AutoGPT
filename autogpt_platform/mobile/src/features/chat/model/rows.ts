import {
  asNumber,
  asRecord,
  asString,
  parseJSONLoose,
} from "../../../lib/json";

export type RowRole = "user" | "assistant" | "reasoning" | "tool";

export interface ToolCall {
  id: string;
  name: string;
  input: unknown;
  displayName: string | null;
}

/**
 * One conversation row, shaped like the backend's persisted ChatMessage.
 * History pages and a live turn both produce rows, so a reload renders
 * exactly what the stream showed.
 */
export interface Row {
  key: string;
  role: RowRole;
  content: string;
  toolCalls: readonly ToolCall[];
  toolCallID: string | null;
  sequence: number | null;
  metadata: Record<string, unknown> | null;
  createdAt: string | null;
}

const ROLES: readonly RowRole[] = ["user", "assistant", "reasoning", "tool"];

/** Rows from `GET /sessions/{id}` messages; unknown roles (e.g. system) are skipped. */
export function rowsFromPersisted(messages: readonly unknown[]): Row[] {
  const rows: Row[] = [];
  messages.forEach((raw, index) => {
    const record = asRecord(raw);
    const role = ROLES.find((r) => r === record?.role);
    if (!record || !role) return;
    const sequence = asNumber(record.sequence);
    rows.push({
      key:
        sequence !== null
          ? `seq:${sequence}`
          : `idx:${index}:${asString(record.id) ?? ""}`,
      role,
      content: typeof record.content === "string" ? record.content : "",
      toolCalls: Array.isArray(record.tool_calls)
        ? record.tool_calls.map(persistedToolCall)
        : [],
      toolCallID: asString(record.tool_call_id),
      sequence,
      metadata: asRecord(record.metadata),
      createdAt: asString(record.created_at),
    });
  });
  return rows;
}

/** Union of loaded pages, ordered by sequence; a later copy of a row wins. */
export function mergeRows(older: readonly Row[], newer: readonly Row[]): Row[] {
  const bySequence = new Map<number, Row>();
  const unsequenced: Row[] = [];
  for (const row of [...older, ...newer]) {
    if (row.sequence === null) unsequenced.push(row);
    else bySequence.set(row.sequence, row);
  }
  const ordered = [...bySequence.values()].sort(
    (a, b) => (a.sequence ?? 0) - (b.sequence ?? 0),
  );
  return [...ordered, ...unsequenced];
}

function persistedToolCall(raw: unknown): ToolCall {
  const call = asRecord(raw);
  const fn = asRecord(call?.function);
  const args = fn?.arguments;
  return {
    id: String(call?.id ?? ""),
    name: String(fn?.name ?? ""),
    input:
      typeof args === "string"
        ? args.trim()
          ? parseJSONLoose(args)
          : {}
        : (args ?? {}),
    displayName:
      typeof call?.display_name === "string" ? call.display_name : null,
  };
}
