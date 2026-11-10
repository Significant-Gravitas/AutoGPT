import type { ToolState, TurnLog } from "./turnLog";

/**
 * A tool call's input as it streams (`tool_input_stream.py`): deltas grow the
 * call's argument text until `tool-input-available` carries the whole input,
 * and `tool-input-error` ends a call whose input never completed. Neither
 * adds a row, since the backend persists a call only once its input is whole.
 *
 * Each returns the next log, or the reason it is a protocol error.
 */
export function extendToolInput(
  log: TurnLog,
  toolCallId: string,
  delta: string,
): TurnLog | string {
  const known = log.tools[toolCallId];
  if (!known) return "unknown tool call";
  if (known.phase !== "input-streaming") return "input closed";
  return withTool(log, toolCallId, {
    ...known,
    inputText: (known.inputText ?? "") + delta,
  });
}

export function failToolInput(
  log: TurnLog,
  toolCallId: string,
  toolName: string,
  errorText: string,
): TurnLog | string {
  const known = log.tools[toolCallId];
  if (known && known.phase !== "input-streaming") return "input closed";
  return withTool(log, toolCallId, {
    ...known,
    name: toolName || known?.name || "",
    phase: "input-error",
    errorText,
  });
}

function withTool(log: TurnLog, id: string, tool: ToolState): TurnLog {
  return { ...log, tools: { ...log.tools, [id]: tool } };
}
