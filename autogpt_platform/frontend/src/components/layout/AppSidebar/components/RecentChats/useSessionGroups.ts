import { getV2ListSessionGroups } from "@/app/api/__generated__/endpoints/chat/chat";
import { SESSION_GROUPS_QUERY_KEY } from "@/app/(platform)/copilot/sessionGroups";
import { shareSessionGroupsData } from "@/app/(platform)/copilot/sessionListSharing";
import {
  SESSION_LIST_REFETCH_INTERVAL_MS,
  useDeletingSessionIds,
} from "@/app/(platform)/copilot/useSessionList";
import { useQuery } from "@tanstack/react-query";
import {
  GROUP_MIN_CHATS,
  SIDEBAR_CHAT_LIMIT,
  withoutDeletingSessions,
} from "./helpers";

/** Every expert the user has chats with, Otto included, each with its real
 *  chat count: at least its newest few chats, with the newest of the rest
 *  filling the sidebar up to its limit, most recently active group first.
 *  Plus the pinned chats on their own when the sidebar shows them in their
 *  own section. One request, refreshed every ten seconds like the list it
 *  replaces. */
export function useSessionGroups(pinnedSeparately: boolean) {
  const query = useQuery({
    queryKey: [...SESSION_GROUPS_QUERY_KEY, { pinnedSeparately }],
    queryFn: () =>
      getV2ListSessionGroups({
        min_per_group: GROUP_MIN_CHATS,
        limit: SIDEBAR_CHAT_LIMIT,
        pinned_separately: pinnedSeparately,
      }),
    structuralSharing: shareSessionGroupsData,
    refetchInterval: SESSION_LIST_REFETCH_INTERVAL_MS,
  });
  const deletingIds = useDeletingSessionIds();
  const data = query.data?.status === 200 ? query.data.data : null;

  return {
    groups: withoutDeletingSessions(data?.groups ?? [], deletingIds),
    pinnedSessions: (data?.pinned ?? []).filter(
      (session) => !deletingIds.includes(session.id),
    ),
    isLoading: query.isLoading,
    // Only a read that never succeeded counts: a failed poll keeps showing
    // the chats it already has.
    loadError: !data && query.isError ? query.error : null,
    retryLoad: query.refetch,
  };
}
