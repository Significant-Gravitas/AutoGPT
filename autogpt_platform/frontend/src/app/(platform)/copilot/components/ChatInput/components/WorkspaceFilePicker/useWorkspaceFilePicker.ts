import { listWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import {
  type InfiniteData,
  keepPreviousData,
  useInfiniteQuery,
} from "@tanstack/react-query";
import { useEffect, useState } from "react";

const SEARCH_DEBOUNCE_MS = 250;
const PAGE_SIZE = 50;

type ListPage = Awaited<ReturnType<typeof listWorkspaceFiles>>;

export function useWorkspaceFilePicker({ enabled }: { enabled: boolean }) {
  const [searchTerm, setSearchTerm] = useState("");
  // Keep the full item (not just id) so a selection survives a search that
  // pages the file off the currently-loaded list.
  const [selected, setSelected] = useState<Map<string, WorkspaceFileItem>>(
    new Map(),
  );

  const debouncedSearch = useDebouncedValue(
    searchTerm.trim(),
    SEARCH_DEBOUNCE_MS,
  );
  const q = debouncedSearch || undefined;

  const query = useInfiniteQuery({
    queryKey: ["workspace-file-picker", "list", { q: q ?? null }] as const,
    queryFn: ({ pageParam }) =>
      listWorkspaceFiles({ limit: PAGE_SIZE, offset: pageParam, q }),
    initialPageParam: 0,
    getNextPageParam: (lastPage, allPages) => {
      if (lastPage.status !== 200) return undefined;
      if (!lastPage.data.has_more) return undefined;
      return countLoadedFiles(allPages);
    },
    placeholderData: keepPreviousData,
    enabled,
  });

  function toggle(item: WorkspaceFileItem) {
    setSelected((prev) => {
      const next = new Map(prev);
      if (next.has(item.id)) {
        next.delete(item.id);
      } else {
        next.set(item.id, item);
      }
      return next;
    });
  }

  function reset() {
    setSelected(new Map());
    setSearchTerm("");
  }

  return {
    files: flattenFiles(query.data),
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    searchTerm,
    setSearchTerm,
    hasMore: !!query.hasNextPage,
    isLoadingMore: query.isFetchingNextPage,
    loadMore: () => {
      query.fetchNextPage();
    },
    selectedIds: selected,
    selectedFiles: Array.from(selected.values()),
    toggle,
    reset,
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
