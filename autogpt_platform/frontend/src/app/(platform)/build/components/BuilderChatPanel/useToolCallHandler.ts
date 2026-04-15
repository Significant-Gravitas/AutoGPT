import { useEffect, useRef } from "react";
import { useChat } from "@ai-sdk/react";
import { parseAsString, useQueryStates } from "nuqs";

type Messages = ReturnType<typeof useChat>["messages"];
type Status = ReturnType<typeof useChat>["status"];

interface UseToolCallHandlerArgs {
  messages: Messages;
  status: Status;
  flowID: string | null;
  onGraphEdited?: () => void;
}

/**
 * Detects completed `edit_agent` and `run_agent` tool calls in assistant messages
 * and dispatches the appropriate side-effects:
 *
 * - `edit_agent` → calls `onGraphEdited` to trigger a graph reload.
 * - `run_agent`  → updates `flowExecutionID` in the URL to auto-follow the new run.
 *
 * Uses `processedToolCallsRef` to deduplicate calls when the messages array
 * updates multiple times while `status === "ready"`.
 * The `flowID` dependency guards against stale callbacks firing after navigation.
 */
export function useToolCallHandler({
  messages,
  status,
  flowID,
  onGraphEdited,
}: UseToolCallHandlerArgs) {
  const [, setQueryStates] = useQueryStates({
    flowID: parseAsString,
    flowExecutionID: parseAsString,
  });

  // Tracks tool call IDs already handled to avoid firing callbacks twice when
  // the messages array updates while status is "ready".
  const processedToolCallsRef = useRef(new Set<string>());

  // Reset the processed set on graph navigation so tool calls from the prior
  // graph are not confused with those from the new graph.
  useEffect(() => {
    processedToolCallsRef.current = new Set();
  }, [flowID]);

  useEffect(() => {
    if (status !== "ready") return;
    for (const msg of messages) {
      if (msg.role !== "assistant") continue;
      for (const part of msg.parts ?? []) {
        if (part.type !== "dynamic-tool") continue;
        const dynPart = part as {
          type: "dynamic-tool";
          toolName: string;
          toolCallId: string;
          state: string;
          output?: unknown;
        };
        if (dynPart.state !== "output-available") continue;
        if (processedToolCallsRef.current.has(dynPart.toolCallId)) continue;
        processedToolCallsRef.current.add(dynPart.toolCallId);

        if (dynPart.toolName === "edit_agent") {
          onGraphEdited?.();
        } else if (dynPart.toolName === "run_agent") {
          const output = dynPart.output as Record<string, unknown> | null;
          const execId = output?.execution_id;
          if (typeof execId === "string" && /^[\w-]+$/i.test(execId)) {
            setQueryStates({ flowExecutionID: execId });
          }
        }
      }
    }
  }, [messages, status, onGraphEdited, setQueryStates]);

  return { processedToolCallsRef };
}
