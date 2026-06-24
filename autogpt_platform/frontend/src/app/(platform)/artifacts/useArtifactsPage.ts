import { useEffect, useState } from "react";
import { listWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { type InfiniteData, useInfiniteQuery } from "@tanstack/react-query";

export type OriginFilter = "all" | "uploaded" | "generated";

const SEARCH_DEBOUNCE_MS = 250;
const ARTIFACTS_PAGE_SIZE = 50;

export const ARTIFACTS_LIST_QUERY_KEY = ["artifacts", "list"] as const;

type ListPage = Awaited<ReturnType<typeof listWorkspaceFiles>>;

export function useArtifactsPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [originFilter, setOriginFilter] = useState<OriginFilter>("all");

  const debouncedSearch = useDebouncedValue(
    searchTerm.trim(),
    SEARCH_DEBOUNCE_MS,
  );

  const q = debouncedSearch || undefined;
  const origin = originFilter === "all" ? undefined : originFilter;

  const query = useInfiniteQuery({
    queryKey: [
      ...ARTIFACTS_LIST_QUERY_KEY,
      { q: q ?? null, origin: origin ?? null },
    ] as const,
    queryFn: ({ pageParam }) =>
      listWorkspaceFiles({
        limit: ARTIFACTS_PAGE_SIZE,
        offset: pageParam,
        q,
        origin,
      }),
    initialPageParam: 0,
    getNextPageParam: (lastPage, allPages) => {
      if (lastPage.status !== 200) return undefined;
      if (!lastPage.data.has_more) return undefined;
      return countLoadedFiles(allPages);
    },
    // No keepPreviousData: switching tabs/search must not flash the previous
    // filter's files. Without it, an uncached filter shows the loading
    // skeleton (isLoading) until its real results arrive; a cached filter
    // still renders instantly from cache.
  });

  return {
    files: flattenFiles(query.data),
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    searchTerm,
    setSearchTerm,
    debouncedSearch,
    originFilter,
    setOriginFilter,
    hasMore: !!query.hasNextPage,
    isLoadingMore: query.isFetchingNextPage,
    loadMore: () => {
      query.fetchNextPage();
    },
  };
}

function flattenFiles(
  data: InfiniteData<ListPage> | undefined,
): WorkspaceFileItem[] {
  if (!data) return [];
  return data.pages.flatMap((page) =>
    page.status === 200 ? (page.data.files ?? []) : [],
  );
}

function countLoadedFiles(pages: ListPage[]): number {
  return pages.reduce(
    (acc, page) =>
      acc + (page.status === 200 ? (page.data.files?.length ?? 0) : 0),
    0,
  );
}

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const handle = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(handle);
  }, [value, delayMs]);
  return debounced;
}
