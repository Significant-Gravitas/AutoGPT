import { getV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { SESSION_GROUP_PAGES_QUERY_KEY } from "@/app/(platform)/copilot/sessionGroups";
import { shareSessionListData } from "@/app/(platform)/copilot/sessionListSharing";
import {
  flattenSessions,
  getNextSessionListOffset,
  useDeletingSessionIds,
} from "@/app/(platform)/copilot/useSessionList";
import { useInfiniteQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import {
  GROUP_PAGE_LIMIT,
  GROUP_PREVIEW_COUNT,
  GROUP_SHOW_MORE_STEP,
  isGroupCollapsed,
  mergeGroupSessions,
  setGroupCollapsed,
} from "../../helpers";

interface Args {
  /** The group's expert; null for Otto's own chats. */
  expertId: string | null;
  total: number;
  previewSessions: SessionSummaryResponse[];
  pinnedSeparately: boolean;
  activeSessionId: string | null;
}

export function useExpertChatGroup({
  expertId,
  total,
  previewSessions,
  pinnedSeparately,
  activeSessionId,
}: Args) {
  const [isOpen, setIsOpen] = useState(() => !isGroupCollapsed(expertId));
  const [visibleCount, setVisibleCount] = useState(GROUP_PREVIEW_COUNT);
  const olderPages = useOlderPages({
    expertId,
    pinnedSeparately,
    enabled: visibleCount > GROUP_PREVIEW_COUNT,
  });
  const deletingIds = useDeletingSessionIds();
  const sessions = mergeGroupSessions(
    previewSessions,
    flattenSessions(olderPages.data),
  ).filter((session) => !deletingIds.includes(session.id));

  // The first "Show more" turns the older pages on, which reads their first
  // page by itself; after that, the next page is read whenever the ones
  // loaded hold fewer chats than the group is asked to show.
  const needsNextPage =
    olderPages.data !== undefined &&
    olderPages.hasNextPage &&
    !olderPages.isFetchingNextPage &&
    sessions.length < visibleCount;
  const { fetchNextPage } = olderPages;
  useEffect(() => {
    if (needsNextPage) void fetchNextPage();
  }, [needsNextPage, fetchNextPage]);

  function changeOpen(open: boolean) {
    setIsOpen(open);
    setGroupCollapsed(expertId, !open);
  }

  function showMore() {
    setVisibleCount((count) => count + GROUP_SHOW_MORE_STEP);
  }

  return {
    isOpen,
    setIsOpen: changeOpen,
    visibleSessions: sessions.slice(0, visibleCount),
    // A collapsed group still shows where the user is and what is running.
    collapsedSessions: sessions.filter(
      (session) => session.is_processing || session.id === activeSessionId,
    ),
    hasMore: visibleCount < total,
    isLoadingMore: olderPages.isLoading || olderPages.isFetchingNextPage,
    showMore,
  };
}

interface OlderPagesArgs {
  expertId: string | null;
  pinnedSeparately: boolean;
  enabled: boolean;
}

/** The group's chats a page at a time, newest first like its preview, for
 *  "Show more": the expert's, or Otto's own when there is no expert. */
function useOlderPages({
  expertId,
  pinnedSeparately,
  enabled,
}: OlderPagesArgs) {
  return useInfiniteQuery({
    queryKey: [
      ...SESSION_GROUP_PAGES_QUERY_KEY,
      expertId,
      { pinnedSeparately },
    ],
    queryFn: ({ pageParam }) =>
      getV2ListSessions({
        limit: GROUP_PAGE_LIMIT,
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
