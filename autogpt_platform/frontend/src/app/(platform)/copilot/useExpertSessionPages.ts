import { getV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import { useInfiniteQuery } from "@tanstack/react-query";
import { SESSION_GROUP_PAGES_QUERY_KEY } from "./sessionGroups";
import { shareSessionListData } from "./sessionListSharing";
import { getNextSessionListOffset } from "./useSessionList";

/** Chats per request; above the sidebar's limit, so a group's first "Show
 *  more" needs one request. */
export const EXPERT_SESSION_PAGE_LIMIT = 40;

interface Args {
  /** The expert; null for Otto's own chats. */
  expertId: string | null;
  /** Leave the pinned chats out, for a list that shows them elsewhere. */
  pinnedSeparately: boolean;
  enabled: boolean;
}

/** One expert's chats a page at a time, newest first: the expert's, or
 *  Otto's own when there is no expert. The sidebar's "Show more" and Home's
 *  recent chats read the same pages. */
export function useExpertSessionPages({
  expertId,
  pinnedSeparately,
  enabled,
}: Args) {
  return useInfiniteQuery({
    queryKey: [
      ...SESSION_GROUP_PAGES_QUERY_KEY,
      expertId,
      { pinnedSeparately },
    ],
    queryFn: ({ pageParam }) =>
      getV2ListSessions({
        limit: EXPERT_SESSION_PAGE_LIMIT,
        offset: pageParam,
        pinned_first: false,
        ...(expertId ? { expert_id: expertId } : { autopilot_only: true }),
        ...(pinnedSeparately ? { pinned: false } : {}),
      }),
    initialPageParam: 0,
    getNextPageParam: getNextSessionListOffset,
    structuralSharing: shareSessionListData,
    enabled,
  });
}
