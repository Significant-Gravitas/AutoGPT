import { useEffect, useState } from "react";
import { listWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { type InfiniteData, useInfiniteQuery } from "@tanstack/react-query";
import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";

export type OriginFilter = "all" | "uploaded" | "generated";

const SEARCH_DEBOUNCE_MS = 250;
const ARTIFACTS_PAGE_SIZE = 50;

export const ARTIFACTS_LIST_QUERY_KEY = ["artifacts", "list"] as const;

type ListPage = Awaited<ReturnType<typeof listWorkspaceFiles>>;

export function useArtifactsPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [originFilter, setOriginFilter] = useState<OriginFilter>("all");
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);
  const [selectedFolderScope, setSelectedFolderScope] = useState<{
    organizationId: string | null;
    teamId: string | null;
  } | null>(null);
  const activeOrgID = useOrgTeamStore((s) => s.activeOrgID);
  const activeTeamID = useOrgTeamStore((s) => s.activeTeamID);
  const isTenantReady = useOrgTeamStore((s) => s.isLoaded);
  const organizationId = selectedFolderScope
    ? selectedFolderScope.organizationId
    : activeOrgID;
  const teamId = selectedFolderScope
    ? selectedFolderScope.teamId
    : activeTeamID;

  const debouncedSearch = useDebouncedValue(
    searchTerm.trim(),
    SEARCH_DEBOUNCE_MS,
  );

  const q = debouncedSearch || undefined;
  const origin = originFilter === "all" ? undefined : originFilter;
  // No folder selected → show only root-level files; a folder is selected →
  // scope the listing to that folder.
  const folderId = selectedFolderId ?? undefined;
  // While searching, span the whole workspace (including files inside folders)
  // so global search isn't limited to root-level files.
  const rootOnly = selectedFolderId === null && !q;

  const query = useInfiniteQuery({
    queryKey: [
      ...ARTIFACTS_LIST_QUERY_KEY,
      {
        q: q ?? null,
        origin: origin ?? null,
        folderId: folderId ?? null,
        rootOnly,
        organizationId,
        teamId,
        isTenantReady,
      },
    ] as const,
    queryFn: ({ pageParam }) =>
      listWorkspaceFiles(
        {
          limit: ARTIFACTS_PAGE_SIZE,
          offset: pageParam,
          q,
          origin,
          folder_id: folderId,
          root_only: rootOnly,
        },
        getTenantRequestInit(organizationId, teamId, isTenantReady),
      ),
    enabled: isTenantReady,
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
    selectedFolderId,
    selectFolder: (folder: WorkspaceFolder | null) => {
      setSelectedFolderId(folder?.id ?? null);
      setSelectedFolderScope(
        folder
          ? {
              organizationId: folder.organization_id ?? null,
              teamId: folder.team_id ?? null,
            }
          : null,
      );
    },
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
