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
