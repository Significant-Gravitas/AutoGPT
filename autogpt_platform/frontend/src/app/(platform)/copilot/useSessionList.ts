import {
  getV2ListSessions,
  type getV2ListSessionsResponse,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { type InfiniteData, useInfiniteQuery } from "@tanstack/react-query";

export const SESSION_LIST_PAGE_SIZE = 50;
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

  const sessions = flattenSessions(query.data);
  const total = getTotal(query.data);

  return {
    sessions,
    total,
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

function getTotal(data: SessionListInfiniteData | undefined) {
  const lastPage = data?.pages.at(-1);
  if (!lastPage || lastPage.status !== 200) return 0;
  return lastPage.data.total;
}

// Re-exported so cache walkers don't have to depend on the generated module.
export type { getV2ListSessionsResponse };
