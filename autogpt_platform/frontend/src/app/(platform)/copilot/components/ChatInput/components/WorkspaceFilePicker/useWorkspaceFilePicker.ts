import {
  listWorkspaceFiles,
  useListWorkspaceFolders,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { okData } from "@/app/api/helpers";
import {
  ancestorsOf,
  childrenOf,
} from "@/app/(platform)/artifacts/components/WorkspaceFolders/folderTree";
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

/** A picked row: a file, or a folder the model will open for itself. */
export type PickedItem =
  | { kind: "file"; file: WorkspaceFileItem }
  | PickedFolder;

type PickedFolder = {
  kind: "folder";
  folder: WorkspaceFolder;
  subfolderCount: number;
};

interface Args {
  enabled: boolean;
  /** Expert the chat is scoped to; lists only files that expert can attach. */
  expertId?: string | null;
}

export function useWorkspaceFilePicker({ enabled, expertId }: Args) {
  const [searchTerm, setSearchTerm] = useState("");
  // Deliberately outside `reset()`: the filter is a property of this chat, so
  // it survives close-and-reopen. The picker's `key={expertId}` remount is
  // what puts it back to ON when the chat's expert changes.
  const [expertOnly, setExpertOnly] = useState(true);
  const [folderId, setFolderId] = useState<string | null>(null);
  // Keep the full item (not just id) so a selection survives a search that
  // pages the row off the currently-loaded list. The anchor travels with the
  // files because a range is only meaningful against the list it was taken from.
  const [selection, setSelection] = useState<Selection>({
    selected: new Map(),
    anchor: null,
  });
  const [pickedFolders, setPickedFolders] = useState<Map<string, PickedFolder>>(
    new Map(),
  );

  const debouncedSearch = useDebouncedValue(
    searchTerm.trim(),
    SEARCH_DEBOUNCE_MS,
  );
  const q = debouncedSearch || undefined;
  // An expert chat with the filter on lists that expert's conversations flat,
  // with no folder axis: the user's own folders are exactly what it hides.
  const isExpertOnly = !!expertId && expertOnly;
  const includeUserFiles = expertId ? !expertOnly : undefined;
  const showFolders = !isExpertOnly && !q;
  // While searching, span every folder so a search never has to be repeated
  // per folder — as the Files page does.
  const rootOnly = showFolders && folderId === null;
  // One value for both the key and the request: the lint rule reads them
  // syntactically, and a key that names less than the query does goes stale.
  const listedFolderId = showFolders ? folderId : null;

  const foldersQuery = useListWorkspaceFolders({
    query: { select: okData, enabled: enabled && !isExpertOnly },
  });
  const folders = foldersQuery.data?.folders ?? [];

  const query = useInfiniteQuery({
    queryKey: [
      "workspace-file-picker",
      "list",
      {
        q: q ?? null,
        expertId: expertId ?? null,
        includeUserFiles: includeUserFiles ?? null,
        folderId: listedFolderId,
        rootOnly,
      },
    ] as const,
    queryFn: ({ pageParam }) =>
      listWorkspaceFiles({
        limit: PAGE_SIZE,
        offset: pageParam,
        q,
        expert_id: expertId ?? undefined,
        include_user_files: includeUserFiles,
        folder_id: listedFolderId ?? undefined,
        root_only: rootOnly || undefined,
      }),
    initialPageParam: 0,
    getNextPageParam: (lastPage, allPages) => {
      if (lastPage.status !== 200) return undefined;
      if (!lastPage.data.has_more) return undefined;
      return countLoadedFiles(allPages);
    },
    // Keep the previous page while a search refines the same listing, but
    // never show one expert's files, the other side of the filter or another
    // folder's files while its request is pending.
    placeholderData: (previousData, previousQuery) => {
      const prev = previousQuery?.queryKey[2];
      if (!prev) return undefined;
      if (prev.expertId !== (expertId ?? null)) return undefined;
      if (prev.includeUserFiles !== (includeUserFiles ?? null))
        return undefined;
      // A search refines a listing; opening a folder replaces it.
      const isBrowsing = prev.q === null && !q;
      if (
        isBrowsing &&
        (prev.folderId !== listedFolderId || prev.rootOnly !== rootOnly)
      )
        return undefined;
      return previousData;
    },
    enabled,
  });

  const files = flattenFiles(query.data);

  function select(index: number, modifiers?: SelectionModifiers) {
    setSelection((prev) => applySelection(prev, files, index, modifiers));
  }

  function toggleFolder(folder: WorkspaceFolder, subfolderCount: number) {
    setPickedFolders((prev) => {
      const next = new Map(prev);
      if (next.has(folder.id)) next.delete(folder.id);
      else next.set(folder.id, { kind: "folder", folder, subfolderCount });
      return next;
    });
  }

  // A range across a changed listing would span files never shown together.
  function dropAnchor() {
    setSelection((prev) => ({ ...prev, anchor: null }));
  }

  function search(term: string) {
    setSearchTerm(term);
    dropAnchor();
  }

  function openFolder(id: string | null) {
    setFolderId(id);
    dropAnchor();
  }

  function filterToExpert(on: boolean) {
    setExpertOnly(on);
    dropAnchor();
  }

  function reset() {
    setSelection({ selected: new Map(), anchor: null });
    setSearchTerm("");
    setPickedFolders(new Map());
    setFolderId(null);
  }

  // Folders first, as the picker shows them, so the chips — and at the cap,
  // what survives it — follow the order on screen.
  const selectedFiles = orderByList(selection.selected, files);
  const selectedItems: PickedItem[] = [
    ...pickedFolders.values(),
    ...selectedFiles.map((file) => ({ kind: "file" as const, file })),
  ];

  return {
    files,
    folderRows: showFolders ? childrenOf(folders, folderId) : [],
    folders,
    breadcrumb:
      showFolders && folderId
        ? ancestorsOf(folders, folderId).map((folder) => ({
            id: folder.id,
            name: folder.name,
          }))
        : [],
    folderId,
    openFolder,
    showFolders,
    isFoldersLoading: foldersQuery.isLoading,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    searchTerm,
    setSearchTerm: search,
    expertOnly,
    setExpertOnly: filterToExpert,
    hasMore: !!query.hasNextPage,
    isLoadingMore: query.isFetchingNextPage,
    loadMore: () => {
      query.fetchNextPage();
    },
    selectedIds: selection.selected,
    selectedFolderIds: new Set(pickedFolders.keys()),
    selectedItems,
    selectedFileCount: selectedFiles.length,
    selectedFolderCount: pickedFolders.size,
    select,
    toggleFolder,
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
