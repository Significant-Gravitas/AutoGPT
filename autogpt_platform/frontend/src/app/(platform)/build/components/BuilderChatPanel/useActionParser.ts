import { useMemo } from "react";
import { useChat } from "@ai-sdk/react";
import {
  extractTextFromParts,
  getActionKey,
  parseGraphActions,
} from "./helpers";

type Messages = ReturnType<typeof useChat>["messages"];
type Status = ReturnType<typeof useChat>["status"];

/**
 * Parses structured graph actions from completed assistant messages.
 *
 * Gated on `status === "ready"` so parsing only runs on fully completed turns,
 * never on partial streaming output. Deduplicates actions by content key so
 * identical suggestions across multiple assistant messages appear only once.
 */
export function useActionParser({
  messages,
  status,
}: {
  messages: Messages;
  status: Status;
}) {
  const parsedActions = useMemo(() => {
    if (status !== "ready") return [];
    const seen = new Set<string>();
    return messages
      .filter((m) => m.role === "assistant")
      .flatMap((msg) => parseGraphActions(extractTextFromParts(msg.parts)))
      .filter((action) => {
        const key = getActionKey(action);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
  }, [messages, status]);

  return { parsedActions };
}
