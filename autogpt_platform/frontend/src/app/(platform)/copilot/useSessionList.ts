import { getV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
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

export interface SessionGroup {
  expertId: string | null;
  sessions: SessionSummaryResponse[];
}

export interface SidebarSessions {
  /** Rendered flat above the groups so the backend's pinned-first ordering
   *  survives grouping. Empty unless headers are shown. */
  pinned: SessionSummaryResponse[];
  groups: SessionGroup[];
  /** Headers only earn their space once there is an expert group to
   *  distinguish from Autopilot. */
  showHeaders: boolean;
}

export function groupSessionsByExpert(
  sessions: SessionSummaryResponse[],
): SessionGroup[] {
  const byExpert = new Map<string | null, SessionSummaryResponse[]>();
  for (const session of sessions) {
    const key = session.expert_id ?? null;
    const bucket = byExpert.get(key);
    if (bucket) {
      bucket.push(session);
    } else {
      byExpert.set(key, [session]);
    }
  }
  return [...byExpert.entries()]
    .map(([expertId, grouped]) => ({ expertId, sessions: grouped }))
    .sort((a, b) => {
      if (a.expertId === b.expertId) return 0;
      if (a.expertId === null) return -1;
      if (b.expertId === null) return 1;
      return 0;
    });
}

export function groupSessionsForSidebar({
  sessions,
  floatPinned,
}: {
  sessions: SessionSummaryResponse[];
  floatPinned: boolean;
}): SidebarSessions {
  const groups = groupSessionsByExpert(sessions);
  // With a single group the list reads exactly as the API returned it, so
  // adding headers (and lifting pinned chats out) would only be noise.
  if (groups.length <= 1) return { pinned: [], groups, showHeaders: false };

  const pinned = floatPinned
    ? sessions.filter((session) => !!session.is_pinned)
    : [];
  if (pinned.length === 0) return { pinned, groups, showHeaders: true };

  return {
    pinned,
    groups: groupSessionsByExpert(
      sessions.filter((session) => !session.is_pinned),
    ),
    showHeaders: true,
  };
}

function countLoadedSessions(pages: SessionListPage[]) {
  return pages.reduce(
    (acc, page) => acc + (page.status === 200 ? page.data.sessions.length : 0),
    0,
  );
}
