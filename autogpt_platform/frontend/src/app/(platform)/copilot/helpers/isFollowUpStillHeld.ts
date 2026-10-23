import { getV2GetPendingMessages } from "@/app/api/__generated__/endpoints/chat/chat";

/**
 * Whether the server still holds *text* as a follow-up nothing will answer.
 * A held chip can be stale: another tab's send, or the turn an answered
 * approval starts, reads the queue too, and resending a message already read
 * would have Otto answer it twice.
 */
export async function isFollowUpStillHeld(sessionId: string, text: string) {
  const res = await getV2GetPendingMessages(sessionId);
  if (res.status !== 200) return false;
  return Boolean(res.data.held_reason) && res.data.messages.includes(text);
}
