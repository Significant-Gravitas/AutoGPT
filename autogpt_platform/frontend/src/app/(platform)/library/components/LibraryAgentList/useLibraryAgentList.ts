"use client";

import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { getGetV2ListLibraryAgentsQueryKey } from "@/app/api/__generated__/endpoints/library/library";
import {
  useGetV2ListLibraryFolders,
  usePostV2BulkMoveAgents,
  getGetV2ListLibraryFoldersQueryKey,
} from "@/app/api/__generated__/endpoints/folders/folders";
import type { getV2ListLibraryFoldersResponseSuccess } from "@/app/api/__generated__/endpoints/folders/folders";
import type { LibraryFolder } from "@/app/api/__generated__/models/libraryFolder";
import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import {
  okData,
  getPaginatedTotalCount,
  getPaginationNextPageNumber,
  unpaginate,
} from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useFavoriteAgents } from "../../hooks/useFavoriteAgents";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

interface Props {
  searchTerm: string;
  librarySort: LibraryAgentSort;
  selectedFolderId: string | null;
  onFolderSelect: (folderId: string | null) => void;
  activeTab: string;
}

export function useLibraryAgentList({
  searchTerm,
  librarySort,
  selectedFolderId,
  onFolderSelect,
  activeTab,
}: Props) {
  const isFavoritesTab = activeTab === "favorites";
  const { toast } = useToast();
  const stableQueryClient = getQueryClient();
  const queryClient = useQueryClient();
  const prevSortRef = useRef<LibraryAgentSort | null>(null);

  const [editingFolder, setEditingFolder] = useState<LibraryFolder | null>(
    null,
  );
  const [deletingFolder, setDeletingFolder] = useState<LibraryFolder | null>(
    null,
  );

  const {
    data: agentsQueryData,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: allAgentsLoading,
  } = useGetV2ListLibraryAgentsInfinite(
    {
      page: 1,
      page_size: 20,
      search_term: searchTerm || undefined,
      sort_by: librarySort,
      folder_id: selectedFolderId ?? undefined,
      include_root_only: selectedFolderId === null ? true : undefined,
    },
    {
      query: {
        getNextPageParam: getPaginationNextPageNumber,
      },
    },
  );

  useEffect(() => {
    if (prevSortRef.current !== null && prevSortRef.current !== librarySort) {
      stableQueryClient.resetQueries({
        queryKey: ["/api/library/agents"],
      });
    }
    prevSortRef.current = librarySort;
  }, [librarySort, stableQueryClient]);

  const allAgentsList = agentsQueryData
    ? unpaginate(agentsQueryData, "agents")
    : [];
  const allAgentsCount = getPaginatedTotalCount(agentsQueryData);

  const favoriteAgentsData = useFavoriteAgents({ searchTerm });

  const {
    agentLoading,
    agentCount,
    allAgents: agents,
    hasNextPage: agentsHasNextPage,
    isFetchingNextPage: agentsIsFetchingNextPage,
    fetchNextPage: agentsFetchNextPage,
  } = isFavoritesTab
    ? favoriteAgentsData
    : {
        agentLoading: allAgentsLoading,
        agentCount: allAgentsCount,
        allAgents: allAgentsList,
        hasNextPage: hasNextPage,
        isFetchingNextPage: isFetchingNextPage,
        fetchNextPage: fetchNextPage,
      };

  const { data: rawFoldersData } = useGetV2ListLibraryFolders(undefined, {
    query: { select: okData },
  });

  const foldersData = searchTerm ? undefined : rawFoldersData;

  const { mutate: moveAgentToFolder } = usePostV2BulkMoveAgents({
    mutation: {
      onMutate: async ({ data }) => {
        await queryClient.cancelQueries({
          queryKey: getGetV2ListLibraryFoldersQueryKey(),
        });
        await queryClient.cancelQueries({
          queryKey: getGetV2ListLibraryAgentsQueryKey(),
        });

        const previousFolders =
          queryClient.getQueriesData<getV2ListLibraryFoldersResponseSuccess>({
            queryKey: getGetV2ListLibraryFoldersQueryKey(),
          });

        if (data.folder_id) {
          queryClient.setQueriesData<getV2ListLibraryFoldersResponseSuccess>(
            { queryKey: getGetV2ListLibraryFoldersQueryKey() },
            (old) => {
              if (!old?.data?.folders) return old;
              return {
                ...old,
                data: {
                  ...old.data,
                  folders: old.data.folders.map((f) =>
                    f.id === data.folder_id
                      ? {
                          ...f,
                          agent_count:
                            (f.agent_count ?? 0) + data.agent_ids.length,
                        }
                      : f,
                  ),
                },
              };
            },
          );
        }

        return { previousFolders };
      },
      onError: (_error, _variables, context) => {
        if (context?.previousFolders) {
          for (const [queryKey, data] of context.previousFolders) {
            queryClient.setQueryData(queryKey, data);
          }
        }
        toast({
          title: "Error",
          description: "Failed to move agent. Please try again.",
          variant: "destructive",
        });
      },
      onSettled: () => {
        queryClient.invalidateQueries({
          queryKey: getGetV2ListLibraryFoldersQueryKey(),
        });
        queryClient.invalidateQueries({
          queryKey: getGetV2ListLibraryAgentsQueryKey(),
        });
      },
    },
  });

  function handleAgentDrop(agentId: string, folderId: string) {
    moveAgentToFolder({
      data: {
        agent_ids: [agentId],
        folder_id: folderId,
      },
    });
  }

  const currentFolder = selectedFolderId
    ? foldersData?.folders.find((f) => f.id === selectedFolderId)
    : null;

  const showFolders = !isFavoritesTab && !selectedFolderId;

  function handleFolderDeleted() {
    if (selectedFolderId === deletingFolder?.id) {
      onFolderSelect(null);
    }
  }

  return {
    isFavoritesTab,
    agentLoading,
    agentCount,
    agents,
    hasNextPage: agentsHasNextPage,
    isFetchingNextPage: agentsIsFetchingNextPage,
    fetchNextPage: agentsFetchNextPage,
    foldersData,
    currentFolder,
    showFolders,
    editingFolder,
    setEditingFolder,
    deletingFolder,
    setDeletingFolder,
    handleAgentDrop,
    handleFolderDeleted,
  };
}
