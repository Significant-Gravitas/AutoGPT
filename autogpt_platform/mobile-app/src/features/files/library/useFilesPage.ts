import { useInfiniteQuery, useQuery, useQueryClient } from "@tanstack/react-query";
import { router, useNavigation } from "expo-router";
import { useRef, useState } from "react";
import { BackHandler, type NativeEventSubscription } from "react-native";

import { firstName } from "@/features/marketplace/helpers";
import { avatarSource, avatarTint } from "@/features/team/helpers";
import { expertKeys, listExperts } from "@/lib/api/experts";
import {
  getStorageUsage,
  listFolders,
  listLibraryPage,
  type WorkspaceFile,
  type WorkspaceFolder,
  workspaceKeys,
} from "@/lib/api/workspace";
import { useMountEffect } from "@/lib/useMountEffect";

import type { FileOwner } from "../components/FileRow";
import { visibleFiles } from "../helpers";
import { childFolders, folderPath, libraryFilter, type OwnerChip } from "../libraryHelpers";

const SEARCH_DELAY_MS = 300;

/** The Files page, at the top level (`folderID` null) or inside one folder. */
export function useFilesPage(folderID: string | null) {
  // What's typed shows at once; the list follows a moment later.
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  const searchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [chip, setChip] = useState<OwnerChip>({ kind: "all" });
  const [openFile, setOpenFile] = useState<WorkspaceFile | null>(null);
  const navigation = useNavigation();
  const queryClient = useQueryClient();

  const filter = libraryFilter(chip, folderID, query);
  const files = useInfiniteQuery({
    queryKey: workspaceKeys.library(filter),
    queryFn: ({ pageParam }) => listLibraryPage(filter, pageParam),
    initialPageParam: 0,
    getNextPageParam: (last, pages) =>
      last.has_more && last.files.length > 0
        ? pages.reduce((count, page) => count + page.files.length, 0)
        : undefined,
  });
  const folders = useQuery({ queryKey: workspaceKeys.folders, queryFn: listFolders });
  const storage = useQuery({
    queryKey: workspaceKeys.storage,
    queryFn: getStorageUsage,
    enabled: folderID === null,
  });
  const experts = useQuery({ queryKey: expertKeys.list, queryFn: listExperts });

  const allFolders = folders.data ?? [];
  const path = folderID ? folderPath(allFolders, folderID) : [];
  const team = [...(experts.data ?? [])].sort((a, b) => a.name.localeCompare(b.name));
  const owners = new Map<string, FileOwner>(
    team.map((expert) => [
      expert.id,
      { name: firstName(expert.name), avatar: avatarSource(expert), tint: avatarTint(expert) },
    ]),
  );

  // On Android the back button goes up a folder while a folder is in front.
  // `goUp` reads the folders at press time, so this mount-time listener stays current.
  useMountEffect(() => {
    let backPress: NativeEventSubscription | null = null;
    function handleFocus() {
      backPress?.remove();
      if (!folderID) return;
      backPress = BackHandler.addEventListener("hardwareBackPress", () => {
        goUp();
        return true;
      });
    }
    function handleBlur() {
      backPress?.remove();
      backPress = null;
    }
    if (navigation.isFocused()) handleFocus();
    const offFocus = navigation.addListener("focus", handleFocus);
    const offBlur = navigation.addListener("blur", handleBlur);
    return () => {
      offFocus();
      offBlur();
      backPress?.remove();
      if (searchTimer.current) clearTimeout(searchTimer.current);
    };
  });

  function changeSearch(text: string) {
    setSearch(text);
    if (searchTimer.current) clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => setQuery(text), SEARCH_DELAY_MS);
  }

  function openFolder(id: string | null) {
    router.navigate(id ? `/files/${encodeURIComponent(id)}` : "/files");
  }

  function goUp() {
    const latest = queryClient.getQueryData<WorkspaceFolder[]>(workspaceKeys.folders) ?? [];
    const trail = folderID ? folderPath(latest, folderID) : [];
    const parentID = trail.length > 1 ? trail[trail.length - 2].id : null;
    router.navigate(parentID ? `/files/${encodeURIComponent(parentID)}` : "/files");
  }

  function ownerOf(file: WorkspaceFile): FileOwner | null {
    if (file.expert_id) return owners.get(file.expert_id) ?? null;
    if (file.origin === "uploaded") return { name: "You", avatar: null, tint: null };
    return null;
  }

  function loadMore() {
    if (files.hasNextPage && !files.isFetchingNextPage) void files.fetchNextPage();
  }

  function refresh() {
    void files.refetch();
    void folders.refetch();
    if (folderID === null) void storage.refetch();
  }

  return {
    search,
    changeSearch,
    chip,
    setChip,
    team,
    files: visibleFiles((files.data?.pages ?? []).flatMap((page) => page.files)),
    isLoading: files.isPending || folders.isPending,
    hasFailed: files.isError && !files.data,
    // Folders hide during a search, so their failure only matters outside one.
    haveFoldersFailed: !query.trim() && folders.isError && !folders.data,
    isRefreshing: files.isRefetching && !files.isFetchingNextPage,
    hasMore: !!files.hasNextPage,
    isLoadingMore: files.isFetchingNextPage,
    loadMore,
    refresh,
    // A search spans every folder, so the folders give way to its results.
    folders: query.trim() ? [] : childFolders(allFolders, folderID),
    allFolders,
    path,
    // Asked for a folder the list doesn't have: deleted on the web, say.
    isMissingFolder: !!folderID && folders.isSuccess && path.length === 0,
    storage: storage.data ?? null,
    ownerOf,
    openFolder,
    goUp,
    openFile,
    setOpenFile,
  };
}
