import { getV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { answeredIdsInRows } from "../../cardDecisions";

/** Whether the stored chat already answered any of these cards, so a tab
 *  opened before another one answered shows that answer instead of sending a
 *  second. A failed read counts as no: the chat route refuses a second answer
 *  anyway, this only spares the stale tab from showing one. */
export async function isAnsweredOnServer(
  sessionId: string,
  confirmationIds: string[],
): Promise<boolean> {
  try {
    const response = await getV2GetSession(sessionId);
    if (response.status !== 200) return false;
    const answered = answeredIdsInRows(response.data.messages);
    return confirmationIds.some((id) => answered.has(id));
  } catch {
    return false;
  }
}
