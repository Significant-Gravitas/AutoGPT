import { listWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import {
  applySelection,
  orderByList,
  type Selection,
  type SelectionModifiers,
} from "./helpers";
import { type InfiniteData, useInfiniteQuery } from "@tanstack/react-query";
import { useState } from "react";

const SEARCH_DEBOUNCE_MS = 250;
const PAGE_SIZE = 50;

type ListPage = Awaited<ReturnType<typeof listWorkspaceFiles>>;

interface Args {
  enabled: boolean;
  /** Expert the chat is scoped to; lists only files that expert can attach. */
  expertId?: string | null;
}

export function useWorkspaceFilePicker({ enabled, expertId }: Args) {
  const [searchTerm, setSearchTerm] = useState("");
  // Keep the full item (not just id) so a selection survives a search that
  // pages the file off the currently-loaded list. The anchor travels with it
  // because a range is only meaningful against the list it was taken from.
  const [selection, setSelection] = useState<Selection>({
    selected: new Map(),
    anchor: null,
  });

  const debouncedSearch = useDebouncedValue(
    searchTerm.trim(),
    SEARCH_DEBOUNCE_MS,
  );
  const q = debouncedSearch || undefined;

  const query = useInfiniteQuery({
    queryKey: [
      "workspace-file-picker",
      "list",
      { q: q ?? null, expertId: expertId ?? null },
    ] as const,
    queryFn: ({ pageParam }) =>
      listWorkspaceFiles({
        limit: PAGE_SIZE,
        offset: pageParam,
        q,
        expert_id: expertId ?? undefined,
      }),
    initialPageParam: 0,
    getNextPageParam: (lastPage, allPages) => {
      if (lastPage.status !== 200) return undefined;
      if (!lastPage.data.has_more) return undefined;
      return countLoadedFiles(allPages);
    },
    // Keep the previous page while a search refines the same expert's list,
    // but never show one expert's files while another's request is pending.
    placeholderData: (previousData, previousQuery) =>
      previousQuery?.queryKey[2].expertId === (expertId ?? null)
        ? previousData
        : undefined,
    enabled,
  });

  const files = flattenFiles(query.data);

  function select(index: number, modifiers?: SelectionModifiers) {
    setSelection((prev) => applySelection(prev, files, index, modifiers));
  }

  function search(term: string) {
    setSearchTerm(term);
    // A range across a changed filter would span files never shown together.
    setSelection((prev) => ({ ...prev, anchor: null }));
  }

  function reset() {
    setSelection({ selected: new Map(), anchor: null });
    setSearchTerm("");
  }

  return {
    files,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    searchTerm,
    setSearchTerm: search,
    hasMore: !!query.hasNextPage,
    isLoadingMore: query.isFetchingNextPage,
    loadMore: () => {
      query.fetchNextPage();
    },
    selectedIds: selection.selected,
    selectedFiles: orderByList(selection.selected, files),
    select,
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
