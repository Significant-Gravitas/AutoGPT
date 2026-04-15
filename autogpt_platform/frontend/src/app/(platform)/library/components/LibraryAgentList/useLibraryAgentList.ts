"use client";

import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { getGetV2ListLibraryAgentsQueryKey } from "@/app/api/__generated__/endpoints/library/library";
import {
  useGetV2ListLibraryFolders,
  useGetV2GetFolder,
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
import { useEffect, useMemo, useRef, useState } from "react";
import type { AgentStatusFilter } from "../../types";
import { useGetV1ListAllExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";

const FILTER_EXHAUST_THRESHOLD = 3;

interface Props {
  searchTerm: string;
  librarySort: LibraryAgentSort;
  selectedFolderId: string | null;
  onFolderSelect: (folderId: string | null) => void;
  activeTab: string;
  statusFilter?: AgentStatusFilter;
}

export function useLibraryAgentList({
  searchTerm,
  librarySort,
  selectedFolderId,
  onFolderSelect,
  activeTab,
  statusFilter = "all",
}: Props) {
  const isFavoritesTab = activeTab === "favorites";
  const { toast } = useToast();
  const stableQueryClient = getQueryClient();
  const queryClient = useQueryClient();
  const prevSortRef = useRef<LibraryAgentSort | null>(null);
  const [consecutiveEmptyPages, setConsecutiveEmptyPages] = useState(0);
  const prevFilteredLengthRef = useRef(0);
  const prevAgentsLengthRef = useRef(0);

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

  const { data: rawFoldersData } = useGetV2ListLibraryFolders(
    { parent_id: selectedFolderId ?? undefined },
    {
      query: { select: okData },
    },
  );

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

  const { data: currentFolderData } = useGetV2GetFolder(
    selectedFolderId ?? "",
    {
      query: { select: okData, enabled: !!selectedFolderId },
    },
  );
  const currentFolder = selectedFolderId ? currentFolderData : null;

  const showFolders = !isFavoritesTab;

  const { data: executions } = useGetV1ListAllExecutions({
    query: { select: okData },
  });

  const { activeGraphIds, errorGraphIds, completedGraphIds } = useMemo(() => {
    const active = new Set<string>();
    const errors = new Set<string>();
    const completed = new Set<string>();
    const cutoff = Date.now() - 72 * 60 * 60 * 1000;
    for (const exec of executions ?? []) {
      if (
        exec.status === AgentExecutionStatus.RUNNING ||
        exec.status === AgentExecutionStatus.QUEUED ||
        exec.status === AgentExecutionStatus.REVIEW
      ) {
        active.add(exec.graph_id);
      }
      const endedTs = exec.ended_at
        ? exec.ended_at instanceof Date
          ? exec.ended_at.getTime()
          : new Date(String(exec.ended_at)).getTime()
        : 0;
      if (
        (exec.status === AgentExecutionStatus.FAILED ||
          exec.status === AgentExecutionStatus.TERMINATED) &&
        endedTs > cutoff
      ) {
        errors.add(exec.graph_id);
      }
      if (exec.status === AgentExecutionStatus.COMPLETED && endedTs > cutoff) {
        completed.add(exec.graph_id);
      }
    }
    return {
      activeGraphIds: active,
      errorGraphIds: errors,
      completedGraphIds: completed,
    };
  }, [executions]);

  const filteredAgents = filterAgentsByStatus(
    agents,
    statusFilter,
    activeGraphIds,
    errorGraphIds,
    completedGraphIds,
  );

  useEffect(() => {
    if (statusFilter === "all") {
      setConsecutiveEmptyPages(0);
      prevFilteredLengthRef.current = filteredAgents.length;
      prevAgentsLengthRef.current = agents.length;
      return;
    }

    if (agents.length > prevAgentsLengthRef.current) {
      const newFilteredCount = filteredAgents.length;
      const previousCount = prevFilteredLengthRef.current;

      if (newFilteredCount > previousCount) {
        setConsecutiveEmptyPages(0);
      } else {
        setConsecutiveEmptyPages((prev) => prev + 1);
      }
    }

    prevAgentsLengthRef.current = agents.length;
    prevFilteredLengthRef.current = filteredAgents.length;
  }, [agents.length, filteredAgents.length, statusFilter]);

  useEffect(() => {
    setConsecutiveEmptyPages(0);
    prevFilteredLengthRef.current = 0;
    prevAgentsLengthRef.current = 0;
  }, [statusFilter]);

  const filteredExhausted =
    statusFilter !== "all" && consecutiveEmptyPages >= FILTER_EXHAUST_THRESHOLD;

  // When a filter is active, show the filtered count instead of the API total.
  const displayedCount =
    statusFilter === "all" ? allAgentsCount : filteredAgents.length;

  function handleFolderDeleted() {
    if (selectedFolderId === deletingFolder?.id) {
      onFolderSelect(null);
    }
  }

  return {
    isFavoritesTab,
    agentLoading,
    agentCount,
    allAgentsCount,
    displayedCount,
    favoritesCount: favoriteAgentsData.agentCount,
    agents: filteredAgents,
    hasNextPage: agentsHasNextPage && !filteredExhausted,
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

function filterAgentsByStatus<
  T extends {
    graph_id: string;
    has_external_trigger: boolean;
    recommended_schedule_cron?: string | null;
  },
>(
  agents: T[],
  statusFilter: AgentStatusFilter,
  activeGraphIds: Set<string>,
  errorGraphIds: Set<string>,
  completedGraphIds: Set<string>,
): T[] {
  if (statusFilter === "all") return agents;
  return agents.filter((agent) => {
    const isRunning = activeGraphIds.has(agent.graph_id);
    const hasError = errorGraphIds.has(agent.graph_id);

    if (statusFilter === "running") return isRunning;
    if (statusFilter === "attention") return hasError && !isRunning;
    if (statusFilter === "completed")
      return completedGraphIds.has(agent.graph_id);
    if (statusFilter === "listening")
      return !isRunning && !hasError && agent.has_external_trigger;
    if (statusFilter === "scheduled")
      return (
        !isRunning &&
        !hasError &&
        !agent.has_external_trigger &&
        !!agent.recommended_schedule_cron
      );
    if (statusFilter === "idle")
      return (
        !isRunning &&
        !hasError &&
        !agent.has_external_trigger &&
        !agent.recommended_schedule_cron
      );
    if (statusFilter === "healthy") return !hasError;
    return true;
  });
}
