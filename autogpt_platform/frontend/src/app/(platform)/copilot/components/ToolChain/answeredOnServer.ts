import { getV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { answeredIdsInRows } from "../../cardDecisions";

/** The cards the stored chat already answered, so a tab opened before another
 *  one answered sends only what is still open. Null when the chat could not be
 *  read: the card then sends as it would, and the chat route still refuses a
 *  second answer; this only spares the stale tab from showing one. */
export async function answeredIdsOnServer(
  sessionId: string,
): Promise<Set<string> | null> {
  try {
    const response = await getV2GetSession(sessionId);
    if (response.status !== 200) return null;
    return answeredIdsInRows(response.data.messages);
  } catch {
    return null;
  }
}
