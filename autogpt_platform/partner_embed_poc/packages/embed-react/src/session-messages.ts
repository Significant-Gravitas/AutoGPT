import type { UIMessage } from "ai";

import type { EmbedRawMessage } from "./session-api";

interface StoredToolCall {
  id?: unknown;
  type?: unknown;
  function?: {
    name?: unknown;
    arguments?: unknown;
  };
  name?: unknown;
  arguments?: unknown;
}

export function persistedMessagesToUI(rows: EmbedRawMessage[]): UIMessage[] {
  const outputs = new Map<string, unknown>();
  for (const row of rows) {
    if (row.role === "tool" && row.tool_call_id) {
      outputs.set(row.tool_call_id, parseJSON(row.content));
    }
  }

  const messages: UIMessage[] = [];
  for (const [index, row] of rows.entries()) {
    if (!["user", "assistant", "reasoning"].includes(row.role)) continue;
    const parts: UIMessage["parts"] = [];
    if (row.role === "reasoning" && row.content) {
      parts.push({ type: "reasoning", text: row.content, state: "done" });
    } else if (row.content) {
      parts.push({ type: "text", text: row.content });
    }
    if (row.role === "assistant" && Array.isArray(row.tool_calls)) {
      for (const call of row.tool_calls as StoredToolCall[]) {
        const toolCallID = stringValue(call.id);
        const toolName = stringValue(call.function?.name ?? call.name, "tool");
        const input = parseJSON(call.function?.arguments ?? call.arguments);
        const output = outputs.get(toolCallID);
        parts.push(
          output === undefined
            ? {
                type: "dynamic-tool",
                toolName,
                toolCallId: toolCallID,
                state: "input-available",
                input,
              }
            : {
                type: "dynamic-tool",
                toolName,
                toolCallId: toolCallID,
                state: "output-available",
                input,
                output,
              },
        );
      }
    }
    if (parts.length === 0) continue;
    messages.push({
      id: row.id || `persisted-${row.sequence ?? index}`,
      role: row.role === "user" ? "user" : "assistant",
      parts,
    });
  }
  return messages;
}

function parseJSON(value: unknown): unknown {
  if (typeof value !== "string") return value ?? {};
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function stringValue(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}
