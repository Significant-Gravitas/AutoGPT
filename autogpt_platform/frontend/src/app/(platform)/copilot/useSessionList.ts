import { getV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import {
  type InfiniteData,
  type QueryClient,
  useInfiniteQuery,
  useMutationState,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useMemo } from "react";
import { shareSessionListData } from "./sessionListSharing";

export const SESSION_LIST_PAGE_SIZE = 50;
// `refetchInterval` on a `useInfiniteQuery` refetches every loaded page, and
// TanStack Query v5 removed `refetchPage` so we can't scope it to page 0. So
// the infinite query polls only while it holds a single page; once older
// pages are loaded, `SESSION_LIST_FIRST_PAGE_QUERY_KEY` polls page 0 alone and
// patches it into the cache. Older pages refresh on focus, on the WebSocket
// invalidations in `useCopilotNotifications`, or when page 0's membership
// changes (which shifts every later offset, so they all have to be refetched).
export const SESSION_LIST_REFETCH_INTERVAL_MS = 10_000;

// Fresh, paginated-cache key. The orval-generated key targets the non-infinite
// `useQuery` cache shape; keeping the infinite cache on a separate key avoids
// shape collisions and lets us hand a stable key to invalidation callsites.
export const SESSION_LIST_QUERY_KEY = ["copilot", "session-list"] as const;

// Outside `SESSION_LIST_QUERY_KEY`'s prefix, so invalidating the list does not
// also re-run this poll.
const SESSION_LIST_FIRST_PAGE_QUERY_KEY = [
  "copilot",
  "session-list-first-page",
] as const;

// Chats with a delete in flight stay out of every list, whatever a refetch
// returns before the server has removed them.
export const SESSION_DELETE_MUTATION_KEY = [
  "copilot",
  "delete-session",
] as const;

type SessionListPage = Awaited<ReturnType<typeof getV2ListSessions>>;
export type SessionListInfiniteData = InfiniteData<SessionListPage>;

interface Args {
  enabled?: boolean;
}

export function useSessionList({ enabled = true }: Args = {}) {
  const queryClient = useQueryClient();
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
    structuralSharing: shareSessionListData,
    refetchInterval: (listQuery) =>
      hasOlderPages(listQuery.state.data)
        ? false
        : SESSION_LIST_REFETCH_INTERVAL_MS,
    // With older pages loaded the poll only covers page 0, so a focus always
    // refreshes every page, however recently page 0 was patched.
    refetchOnWindowFocus: (listQuery) =>
      hasOlderPages(listQuery.state.data) ? "always" : true,
    enabled,
  });
  const isPollingFirstPage = enabled && hasOlderPages(query.data);
  useQuery({
    // Keyed on whether it polls, so the entry is created (and seeded from the
    // list below) at the moment an older page lands, not at first mount.
    queryKey: [...SESSION_LIST_FIRST_PAGE_QUERY_KEY, isPollingFirstPage],
    queryFn: ({ client }) => refreshFirstPage(client),
    enabled: isPollingFirstPage,
    refetchInterval: SESSION_LIST_REFETCH_INTERVAL_MS,
    // Seeded from the list itself, so loading an older page does not also
    // re-request page 0 straight away; the first poll comes one interval on.
    initialData: () => readFirstPage(queryClient),
    initialDataUpdatedAt: () =>
      queryClient.getQueryState(SESSION_LIST_QUERY_KEY)?.dataUpdatedAt,
    staleTime: SESSION_LIST_REFETCH_INTERVAL_MS,
    // The list's own mount and focus refetches already cover every page, so
    // this query only ever fetches on its interval.
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    // Nothing reads this cache entry; the poll only patches the list.
    notifyOnChangeProps: [],
  });
  const deletingIds = useMutationState({
    filters: { mutationKey: SESSION_DELETE_MUTATION_KEY, status: "pending" },
    select: (mutation) =>
      (mutation.state.variables as { sessionId: string }).sessionId,
  });
  // `query.data` and `deletingIds` keep their identity while nothing in them
  // changes (structural sharing), so a refetch that returns the same sessions
  // hands the lists the same array and every memoised row skips its render.
  const sessions = useMemo(
    () =>
      flattenSessions(query.data).filter(
        (session) => !deletingIds.includes(session.id),
      ),
    [query.data, deletingIds],
  );

  return {
    sessions,
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

/**
 * Puts a freshly fetched page 0 in place of the cached one, keeping the older
 * pages. Returns `null` when page 0 no longer holds the same sessions in the
 * same order (or `total` moved): every later offset has shifted then, so the
 * caller must refetch all loaded pages instead.
 */
export function replaceFirstPage(
  data: SessionListInfiniteData | undefined,
  firstPage: SessionListPage,
): SessionListInfiniteData | null {
  const cached = data?.pages[0];
  if (!data || !cached) return null;
  if (cached.status !== 200 || firstPage.status !== 200) return null;
  const before = cached.data.sessions;
  const after = firstPage.data.sessions;
  if (cached.data.total !== firstPage.data.total) return null;
  if (before.length !== after.length) return null;
  if (before.some((session, i) => session.id !== after[i].id)) return null;
  return { ...data, pages: [firstPage, ...data.pages.slice(1)] };
}

export function removeSessionFromList(
  data: SessionListInfiniteData | undefined,
  sessionId: string,
): SessionListInfiniteData | undefined {
  if (!data) return data;
  return {
    ...data,
    pages: data.pages.map((page) => {
      if (page.status !== 200) return page;
      const sessions = page.data.sessions.filter(
        (session) => session.id !== sessionId,
      );
      if (sessions.length === page.data.sessions.length) return page;
      // Lowering `total` keeps `getNextPageParam`'s offset in step with the
      // server once the row is really gone.
      return {
        ...page,
        data: {
          ...page.data,
          sessions,
          total: Math.max(0, page.data.total - 1),
        },
      };
    }),
  };
}

export interface SessionGroup {
  expertId: string | null;
  sessions: SessionSummaryResponse[];
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

function hasOlderPages(data: SessionListInfiniteData | undefined) {
  return (data?.pages.length ?? 0) > 1;
}

function readFirstPage(queryClient: QueryClient) {
  return queryClient.getQueryData<SessionListInfiniteData>(
    SESSION_LIST_QUERY_KEY,
  )?.pages[0];
}

async function refreshFirstPage(queryClient: QueryClient) {
  const firstPageBefore = readFirstPage(queryClient);
  const firstPage = await getV2ListSessions({
    limit: SESSION_LIST_PAGE_SIZE,
    offset: 0,
  });
  // A list fetch still in flight (say "Load older chats") started from the
  // old pages and would put the old page 0 back when it lands, so patch after
  // it instead.
  await settleListFetch(queryClient);
  // Page 0 changed while this poll was out (a refetch or an edit of the
  // list), so it is at least as new as what this poll brought back. Sharing
  // keeps page 0's object while its content is unchanged, so a fetch of an
  // older page alone does not count.
  if (readFirstPage(queryClient) !== firstPageBefore) return firstPage;
  const next = replaceFirstPage(
    queryClient.getQueryData<SessionListInfiniteData>(SESSION_LIST_QUERY_KEY),
    firstPage,
  );
  if (next) {
    // Keeps the list's own timestamp: the older pages were not fetched, so
    // they must not start counting as fresh.
    queryClient.setQueryData(SESSION_LIST_QUERY_KEY, next, {
      updatedAt: queryClient.getQueryState(SESSION_LIST_QUERY_KEY)
        ?.dataUpdatedAt,
    });
  } else {
    await queryClient.invalidateQueries({ queryKey: SESSION_LIST_QUERY_KEY });
  }
  return firstPage;
}

async function settleListFetch(queryClient: QueryClient) {
  const listQuery = queryClient
    .getQueryCache()
    .find({ queryKey: SESSION_LIST_QUERY_KEY, exact: true });
  if (listQuery?.state.fetchStatus !== "fetching") return;
  await listQuery.promise?.catch(() => undefined);
}

function countLoadedSessions(pages: SessionListPage[]) {
  return pages.reduce(
    (acc, page) => acc + (page.status === 200 ? page.data.sessions.length : 0),
    0,
  );
}
