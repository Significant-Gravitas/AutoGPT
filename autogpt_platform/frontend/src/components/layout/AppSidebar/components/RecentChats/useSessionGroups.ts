import { getV2ListSessionGroups } from "@/app/api/__generated__/endpoints/chat/chat";
import { SESSION_GROUPS_QUERY_KEY } from "@/app/(platform)/copilot/sessionGroups";
import { shareSessionGroupsData } from "@/app/(platform)/copilot/sessionListSharing";
import {
  SESSION_LIST_REFETCH_INTERVAL_MS,
  useDeletingSessionIds,
} from "@/app/(platform)/copilot/useSessionList";
import { useQuery } from "@tanstack/react-query";
import { GROUP_PREVIEW_COUNT, withoutDeletingSessions } from "./helpers";

/** Every expert the user has chats with, Otto included, each with its real
 *  chat count and newest chats, most recently active first; plus the pinned
 *  chats on their own when the sidebar shows them in their own section. One
 *  request, refreshed every ten seconds like the list it replaces. */
export function useSessionGroups(pinnedSeparately: boolean) {
  const query = useQuery({
    queryKey: [...SESSION_GROUPS_QUERY_KEY, { pinnedSeparately }],
    queryFn: () =>
      getV2ListSessionGroups({
        per_group: GROUP_PREVIEW_COUNT,
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
  };
}
