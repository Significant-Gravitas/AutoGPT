import { getV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { SESSION_GROUP_PAGES_QUERY_KEY } from "@/app/(platform)/copilot/sessionGroups";
import { shareSessionListData } from "@/app/(platform)/copilot/sessionListSharing";
import {
  flattenSessions,
  getNextSessionListOffset,
  useDeletingSessionIds,
} from "@/app/(platform)/copilot/useSessionList";
import { useSidebar } from "@/components/molecules/Sidebar/Sidebar";
import { useInfiniteQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import {
  GROUP_PAGE_LIMIT,
  GROUP_SHOW_MORE_STEP,
  isGroupCollapsed,
  mergeGroupSessions,
  olderPagesMatchGroup,
  openingChatCount,
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
  // Chats asked for past the ones the group opens on. The grouped read
  // decides how many that is, and can change it on any poll.
  const [extraCount, setExtraCount] = useState(0);
  const { isMobile } = useSidebar();
  const visibleCount =
    openingChatCount(previewSessions.length, isMobile) + extraCount;
  const olderPages = useOlderPages({
    expertId,
    pinnedSeparately,
    // On a phone the first presses show chats the grouped read already has.
    enabled: visibleCount > previewSessions.length,
  });
  const deletingIds = useDeletingSessionIds();
  const sessions = mergeGroupSessions(
    previewSessions,
    flattenSessions(olderPages.data),
  ).filter((session) => !deletingIds.includes(session.id));
  const visibleSessions = sessions.slice(0, visibleCount);

  useReadPagesToShow(olderPages, sessions.length < visibleCount);
  useRereadShiftedPages({
    olderPages,
    previewSessions,
    total,
    isDeleting: deletingIds.length > 0,
  });

  function changeOpen(open: boolean) {
    setIsOpen(open);
    setGroupCollapsed(expertId, !open);
  }

  // A read that failed is retried by the next press rather than skipped.
  function showMore() {
    if (olderPages.isFetchNextPageError) void olderPages.fetchNextPage();
    else if (olderPages.isError) void olderPages.refetch();
    else setExtraCount((count) => count + GROUP_SHOW_MORE_STEP);
  }

  return {
    isOpen,
    setIsOpen: changeOpen,
    visibleSessions,
    // A collapsed group still shows where the user is and what is running.
    collapsedSessions: sessions.filter(
      (session) => session.is_processing || session.id === activeSessionId,
    ),
    // Counted on the chats on screen, so the button stays until the ones it
    // asked for have arrived, and is there to retry a read that failed.
    hasMore: visibleSessions.length < total,
    isLoadingMore:
      (olderPages.isFetching && olderPages.data === undefined) ||
      olderPages.isFetchingNextPage,
    loadFailed: olderPages.isError && !olderPages.isFetching,
    showMore,
  };
}

type OlderPages = ReturnType<typeof useOlderPages>;

/** The first "Show more" turns the older pages on, which reads their first
 *  page by itself; after that, the next page is read whenever the ones
 *  loaded hold fewer chats than the group is asked to show. */
function useReadPagesToShow(olderPages: OlderPages, wantsMore: boolean) {
  const needsNextPage =
    olderPages.data !== undefined &&
    olderPages.hasNextPage &&
    !olderPages.isFetchingNextPage &&
    !olderPages.isFetchNextPageError &&
    wantsMore;
  const { fetchNextPage } = olderPages;
  useEffect(() => {
    if (needsNextPage) void fetchNextPage();
  }, [needsNextPage, fetchNextPage]);
}

interface RereadArgs {
  olderPages: OlderPages;
  previewSessions: SessionSummaryResponse[];
  total: number;
  isDeleting: boolean;
}

/** Re-reads the older pages, once per change, when the polled total or
 *  newest chats stop matching them: a chat added or deleted since, on
 *  another device say, shifts every offset after it. */
function useRereadShiftedPages({
  olderPages,
  previewSessions,
  total,
  isDeleting,
}: RereadArgs) {
  const { data, isFetching, refetch } = olderPages;
  const groupState = `${total}:${previewSessions.map((s) => s.id).join(",")}`;
  const checkedFor = useRef<string | null>(null);
  useEffect(() => {
    if (!data || isFetching || isDeleting) return;
    if (checkedFor.current === groupState) return;
    checkedFor.current = groupState;
    if (!olderPagesMatchGroup(data, previewSessions, total)) void refetch();
  }, [
    data,
    isFetching,
    isDeleting,
    groupState,
    previewSessions,
    total,
    refetch,
  ]);
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
