import type { ExpertIdentity } from "@/app/(platform)/copilot/useExpertMap";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";

// Fired experts can't be addressed, so their group gets no new-chat entry:
// the deep link would only fall back to Otto. `new=1` keeps the page on the
// fresh new-task state: without it `useChatSession` adopts the expert's
// latest thread, which is exactly the chat the user is trying to leave.
export function getNewChatHref(
  expertId: string | null,
  activeExpertIds: ReadonlySet<string>,
) {
  if (!expertId) return "/home";
  if (!activeExpertIds.has(expertId)) return null;
  return `/home?expertId=${encodeURIComponent(expertId)}&new=1`;
}

// How long a new expert keeps their own sidebar entry while they have no chat.
export const NEW_EXPERT_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;

/** Experts who joined in the last week and have no chat yet, newest first.
 *  Each gets a sidebar group leading to their first chat; once a chat exists
 *  their normal group takes over. Only loaded chats are seen, so an expert
 *  whose chats are all past the first page still gets one, and its link then
 *  opens the chat they already have. */
export function getNewExpertsWithoutChats(
  activeExperts: ExpertIdentity[],
  sessions: SessionSummaryResponse[],
  now: number,
) {
  const withChats = new Set(sessions.map((session) => session.expert_id));
  return activeExperts
    .filter((expert) => {
      const joined = expert.createdAt?.getTime();
      return (
        joined !== undefined &&
        now - joined < NEW_EXPERT_WINDOW_MS &&
        !withChats.has(expert.id)
      );
    })
    .sort(
      (a, b) => (b.createdAt?.getTime() ?? 0) - (a.createdAt?.getTime() ?? 0),
    );
}
