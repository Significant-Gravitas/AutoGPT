import type { UIDataTypes, UIMessage, UITools } from "ai";

import { readPartialToolInput } from "../stream/partialToolInput";
import type { ToolState, TurnLog } from "../stream/turnLog";

type UiPart = UIMessage<unknown, UIDataTypes, UITools>["parts"][number];

export interface PendingTool {
  id: string;
  tool: ToolState;
  displayName: string | undefined;
}

/**
 * Calls whose part has opened but that have no row yet: their input is still
 * streaming, or it never completed. The backend stores a call only once its
 * input is whole, so these live in the log's tool states, not its rows.
 */
export function pendingTools(log: TurnLog): PendingTool[] {
  const withRow = new Set(
    log.rows.flatMap((row) => row.toolCalls.map((call) => call.id)),
  );
  return Object.entries(log.tools)
    .filter(
      ([id, tool]) =>
        !withRow.has(id) &&
        (tool.phase === "input-streaming" || tool.phase === "input-error"),
    )
    .map(([id, tool]) => ({ id, tool, displayName: log.displayNames[id] }));
}

/**
 * A pending call's part, as the AI SDK draws one: `input-streaming` with the
 * input read so far, or `output-error` once its input failed or the turn
 * ended without it.
 */
export function pendingToolPart(
  { id, tool, displayName }: PendingTool,
  turnOver: boolean,
): UiPart {
  const base = {
    type: `tool-${tool.name}` as const,
    toolCallId: id,
    title: displayName,
    input: readPartialToolInput(tool.inputText ?? ""),
  };
  if (tool.phase === "input-error") {
    return {
      ...base,
      state: "output-error",
      errorText: tool.errorText ?? "Interrupted",
    } as UiPart;
  }
  if (turnOver) {
    return {
      ...base,
      state: "output-error",
      errorText: "Interrupted",
    } as UiPart;
  }
  return { ...base, state: "input-streaming" } as UiPart;
}
