import { getV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import { type InfiniteData, useInfiniteQuery } from "@tanstack/react-query";

export const SESSION_LIST_PAGE_SIZE = 50;
// `refetchInterval` on a `useInfiniteQuery` refetches every loaded page, and
// TanStack Query v5 removed `refetchPage` so we can't scope it to page 0.
// Worst case is bounded by the user's session count (and the WebSocket
// invalidations in `useCopilotNotifications` already handle the high-signal
// completion events) — revisit by extracting a separate live-status query if
// this ever becomes a real bandwidth concern.
export const SESSION_LIST_REFETCH_INTERVAL_MS = 10_000;

// Fresh, paginated-cache key. The orval-generated key targets the non-infinite
// `useQuery` cache shape; keeping the infinite cache on a separate key avoids
// shape collisions and lets us hand a stable key to invalidation callsites.
export const SESSION_LIST_QUERY_KEY = ["copilot", "session-list"] as const;

type SessionListPage = Awaited<ReturnType<typeof getV2ListSessions>>;
export type SessionListInfiniteData = InfiniteData<SessionListPage>;

interface Args {
  enabled?: boolean;
}

export function useSessionList({ enabled = true }: Args = {}) {
  const query = useInfiniteQuery({
    queryKey: SESSION_LIST_QUERY_KEY,
    queryFn: ({ pageParam }) =>
      getV2ListSessions({
        limit: SESSION_LIST_PAGE_SIZE,
        offset: pageParam,
      }),
    initialPageParam: 0,
    getNextPageParam: (lastPage, allPages) => {
      if (lastPage.status !== 200) return undefined;
      const loaded = countLoadedSessions(allPages);
      return loaded < lastPage.data.total ? loaded : undefined;
    },
    refetchInterval: SESSION_LIST_REFETCH_INTERVAL_MS,
    enabled,
  });

  return {
    sessions: flattenSessions(query.data),
    isLoading: query.isLoading,
    hasMore: !!query.hasNextPage,
    isLoadingMore: query.isFetchingNextPage,
    loadMore: query.fetchNextPage,
  };
}

export function flattenSessions(data: SessionListInfiniteData | undefined) {
  if (!data) return [];
  return data.pages.flatMap((page) =>
    page.status === 200 ? page.data.sessions : [],
  );
}

function countLoadedSessions(pages: SessionListPage[]) {
  return pages.reduce(
    (acc, page) => acc + (page.status === 200 ? page.data.sessions.length : 0),
    0,
  );
}
