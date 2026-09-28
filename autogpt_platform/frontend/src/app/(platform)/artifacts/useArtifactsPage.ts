import { useEffect, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { listWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { type InfiniteData, useInfiniteQuery } from "@tanstack/react-query";
import { ancestorsOf } from "./components/WorkspaceFolders/folderTree";

export type OriginFilter = "all" | "uploaded" | "generated";
export type ArtifactsView = "list" | "grid";

const SEARCH_DEBOUNCE_MS = 250;
const ARTIFACTS_PAGE_SIZE = 50;
const FOLDER_PARAM = "folder";

export const ARTIFACTS_LIST_QUERY_KEY = ["artifacts", "list"] as const;

type ListPage = Awaited<ReturnType<typeof listWorkspaceFiles>>;

interface Options {
  folders: WorkspaceFolder[];
  isFoldersLoading: boolean;
  isFoldersError: boolean;
}

export function useArtifactsPage({
  folders,
  isFoldersLoading,
  isFoldersError,
}: Options) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [searchTerm, setSearchTerm] = useState("");
  const [originFilter, setOriginFilter] = useState<OriginFilter>("all");
  const [expertFilter, setExpertFilter] = useState<string | null>(null);
  const [view, setView] = useState<ArtifactsView>("list");

  // The open folder lives in the URL so a reload, a shared link and the back
  // button land in it. Only a settled folders query can prove an id is gone,
  // so until then it is taken at face value and the files load straight away
  // rather than queueing behind the folder list.
  const folderParam = searchParams.get(FOLDER_PARAM);
  const isStaleFolder =
    folderParam !== null &&
    !isFoldersLoading &&
    !isFoldersError &&
    !folders.some((folder) => folder.id === folderParam);
  const selectedFolderId = isStaleFolder ? null : folderParam;

  // Remembered while the folder is live, so deleting it (or an ancestor) can
  // land the user one level up instead of back at the root.
  const ancestorIdsRef = useRef<string[]>([]);
  if (selectedFolderId !== null) {
    ancestorIdsRef.current = ancestorsOf(folders, selectedFolderId)
      .slice(0, -1)
      .map((folder) => folder.id);
  }

  const currentSearch = searchParams.toString();
  useEffect(() => {
    if (!isStaleFolder) return;
    const survivingAncestor = [...ancestorIdsRef.current]
      .reverse()
      .find((id) => folders.some((folder) => folder.id === id));
    router.replace(
      folderHref(pathname, currentSearch, survivingAncestor ?? null),
    );
  }, [isStaleFolder, pathname, currentSearch, router, folders]);

  const debouncedSearch = useDebouncedValue(
    searchTerm.trim(),
    SEARCH_DEBOUNCE_MS,
  );

  const q = debouncedSearch || undefined;
  const origin = originFilter === "all" ? undefined : originFilter;
  // "From: <expert>" narrows the files shown where you are, the way "Type"
  // does. `include_user_files` is left unsent, so the tab keeps meaning "made
  // in this expert's chats".
  const expertId = expertFilter ?? undefined;
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
        expertId: expertId ?? null,
      },
    ] as const,
    queryFn: ({ pageParam }) =>
      listWorkspaceFiles({
        limit: ARTIFACTS_PAGE_SIZE,
        offset: pageParam,
        q,
        origin,
        folder_id: folderId,
        root_only: rootOnly,
        expert_id: expertId,
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
    selectedFolderId,
    // Entering a folder is a step forward, so the back button leaves it;
    // leaving replaces, so back does not drop you straight back inside.
    openFolder: (folderId: string) =>
      router.push(folderHref(pathname, currentSearch, folderId)),
    closeFolder: () => {
      if (folderParam === null) return;
      router.replace(folderHref(pathname, currentSearch, null));
    },
    expertFilter,
    setExpertFilter,
    view,
    setView,
    hasMore: !!query.hasNextPage,
    isLoadingMore: query.isFetchingNextPage,
    loadMore: () => {
      query.fetchNextPage();
    },
  };
}

function folderHref(
  pathname: string,
  search: string,
  folderId: string | null,
): string {
  const params = new URLSearchParams(search);
  if (folderId) params.set(FOLDER_PARAM, folderId);
  else params.delete(FOLDER_PARAM);
  const query = params.toString();
  return query ? `${pathname}?${query}` : pathname;
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
