import type { UIDataTypes, UIMessage, UITools } from "ai";
import { getCardDecisions } from "../../cardDecisions";
import { answeredIdsOnServer } from "../ToolChain/answeredOnServer";

/** Whether the stored chat holds an answer to every card this row decided:
 *  saved, or answered from another tab. Asked of the chat itself, since a row
 *  on screen does not say it was saved: a sent row keeps its local id after
 *  its turn fails, and the stream runtime's rows never take a sequence id.
 *  Null when the chat cannot be read, so the caller does not guess. */
export async function isDecisionSavedInChat(
  message: UIMessage<unknown, UIDataTypes, UITools>,
  sessionId: string | null,
): Promise<boolean | null> {
  const ids = (getCardDecisions(message) ?? []).flatMap((decision) =>
    decision.confirmationId ? [decision.confirmationId] : [],
  );
  if (!sessionId || ids.length === 0) return false;
  const answered = await answeredIdsOnServer(sessionId, ids);
  if (answered === null) return null;
  return ids.every((id) => answered.has(id));
}
