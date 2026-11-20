import type { getV2ListSessionGroupsResponse } from "@/app/api/__generated__/endpoints/chat/chat";
import type { QueryClient } from "@tanstack/react-query";
import { SESSION_LIST_QUERY_KEY } from "./useSessionList";

// Both live under the list's key, so every invalidation of the session list
// (a new chat, a rename, a delete, a WebSocket event) refreshes them too.
// "groups" and "group" are distinct segments, so neither key matches the other.

/** The sidebar's read: every expert's group with its real total and newest
 *  chats. */
export const SESSION_GROUPS_QUERY_KEY = [
  ...SESSION_LIST_QUERY_KEY,
  "groups",
] as const;

/** One group's older chats, paged in once its "Show more" is pressed. */
export const SESSION_GROUP_PAGES_QUERY_KEY = [
  ...SESSION_LIST_QUERY_KEY,
  "group",
] as const;

export type SessionGroupsResult = getV2ListSessionGroupsResponse;

/** Whether the sidebar's grouped read already holds a title for this chat. */
export function hasGroupedSessionTitle(
  queryClient: QueryClient,
  sessionId: string,
) {
  return queryClient
    .getQueriesData<SessionGroupsResult>({ queryKey: SESSION_GROUPS_QUERY_KEY })
    .some(
      ([, data]) =>
        data?.status === 200 &&
        [
          ...(data.data.pinned ?? []),
          ...data.data.groups.flatMap((group) => group.sessions),
        ].some((session) => session.id === sessionId && !!session.title),
    );
}

/** Drops a deleted chat from the grouped read and lowers its group's total,
 *  so it doesn't flash back while the refetch after the delete is in flight. */
export function removeSessionFromGroups(
  data: SessionGroupsResult | undefined,
  sessionId: string,
): SessionGroupsResult | undefined {
  if (data?.status !== 200) return data;
  const pinned = data.data.pinned ?? [];
  const inGroup = data.data.groups.some((group) =>
    group.sessions.some((session) => session.id === sessionId),
  );
  const inPinned = pinned.some((session) => session.id === sessionId);
  if (!inGroup && !inPinned) return data;
  const groups = data.data.groups
    .map((group) =>
      group.sessions.some((session) => session.id === sessionId)
        ? {
            ...group,
            total: Math.max(0, group.total - 1),
            sessions: group.sessions.filter(
              (session) => session.id !== sessionId,
            ),
          }
        : group,
    )
    .filter((group) => group.total > 0);
  return {
    ...data,
    data: {
      ...data.data,
      groups,
      pinned: pinned.filter((session) => session.id !== sessionId),
    },
  };
}
