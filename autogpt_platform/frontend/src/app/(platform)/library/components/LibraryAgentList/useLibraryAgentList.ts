"use client";

import {
  getGetV2ListLibraryAgentsQueryKey,
  useGetV2ListLibraryAgentsInfinite,
} from "@/app/api/__generated__/endpoints/library/library";
import {
  getV2ListLibraryFolders,
  useGetV2ListLibraryFolders,
  useGetV2GetFolder,
  postV2BulkMoveAgents,
  getGetV2GetFolderQueryKey,
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
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import type { AgentStatusFilter } from "../../types";
import {
  getGetV1ListAllExecutionsQueryKey,
  useGetV1ListAllExecutions,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { isAgentScheduled } from "../../hooks/executionHelpers";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";
import { getTenantEntityKey } from "@/services/org-team/identity";

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
  const activeOrgID = useOrgTeamStore((s) => s.activeOrgID);
  const activeTeamID = useOrgTeamStore((s) => s.activeTeamID);
  const teams = useOrgTeamStore((s) => s.teams);
  const isTenantReady = useOrgTeamStore((s) => s.isLoaded);
  const [selectedFolderScope, setSelectedFolderScope] = useState<{
    organizationId: string | null;
    teamId: string | null;
  } | null>(null);

  const [editingFolder, setEditingFolder] = useState<LibraryFolder | null>(
    null,
  );
  const [deletingFolder, setDeletingFolder] = useState<LibraryFolder | null>(
    null,
  );

  const agentParams = {
    page: 1,
    page_size: 20,
    search_term: searchTerm || undefined,
    sort_by: librarySort,
    folder_id: selectedFolderId ?? undefined,
    include_root_only: selectedFolderId === null ? true : undefined,
    is_hidden: false,
  };
  const queryOrganizationId = selectedFolderScope
    ? selectedFolderScope.organizationId
    : activeOrgID;
  const queryTeamId = selectedFolderScope
    ? selectedFolderScope.teamId
    : activeTeamID;

  const {
    data: agentsQueryData,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: allAgentsLoading,
  } = useGetV2ListLibraryAgentsInfinite(agentParams, {
    query: {
      enabled: isTenantReady,
      getNextPageParam: getPaginationNextPageNumber,
      queryKey: getTeamScopedQueryKey(
        getGetV2ListLibraryAgentsQueryKey(agentParams),
        queryOrganizationId,
        queryTeamId,
      ),
    },
    request: getTenantRequestInit(
      queryOrganizationId,
      queryTeamId,
      isTenantReady,
    ),
  });

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

  const favoriteAgentsData = useFavoriteAgents({
    searchTerm,
    organizationId: activeOrgID,
    teamId: activeTeamID,
  });

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

  const folderParams = { parent_id: selectedFolderId ?? undefined };
  const organizationTeamIds = useMemo(
    () =>
      teams
        .filter((team) => team.orgId === activeOrgID)
        .map((team) => team.id)
        .sort(),
    [activeOrgID, teams],
  );
  const shouldAggregateOrganizationRootFolders =
    selectedFolderId === null &&
    selectedFolderScope === null &&
    activeOrgID !== null &&
    activeTeamID === null &&
    isTenantReady;
  const { data: exactFoldersData } = useGetV2ListLibraryFolders(folderParams, {
    query: {
      enabled: isTenantReady && !shouldAggregateOrganizationRootFolders,
      queryKey: getTeamScopedQueryKey(
        getGetV2ListLibraryFoldersQueryKey(folderParams),
        queryOrganizationId,
        queryTeamId,
      ),
      select: okData,
    },
    request: getTenantRequestInit(
      queryOrganizationId,
      queryTeamId,
      isTenantReady,
    ),
  });
  const { data: organizationRootFoldersData } = useQuery({
    queryKey: [
      ...getGetV2ListLibraryFoldersQueryKey(folderParams),
      {
        aggregateOrganizationId: activeOrgID,
        teamIds: organizationTeamIds,
      },
    ],
    enabled: shouldAggregateOrganizationRootFolders,
    queryFn: async () => {
      const responses = await Promise.all(
        [null, ...organizationTeamIds].map((teamId) =>
          getV2ListLibraryFolders(
            folderParams,
            getTenantRequestInit(activeOrgID, teamId, true),
          ),
        ),
      );
      const foldersById = new Map<string, LibraryFolder>();

      for (const response of responses) {
        const folderList = okData(response);
        for (const folder of folderList?.folders ?? []) {
          foldersById.set(folder.id, folder);
        }
      }

      const folders = Array.from(foldersById.values());
      return {
        folders,
        pagination: {
          total_items: folders.length,
          total_pages: 1,
          current_page: 1,
          page_size: folders.length,
        },
      };
    },
  });
  const rawFoldersData = shouldAggregateOrganizationRootFolders
    ? organizationRootFoldersData
    : exactFoldersData;

  const foldersData = searchTerm ? undefined : rawFoldersData;

  const { mutate: moveAgentToFolder } = useMutation({
    mutationFn: ({
      data,
      organizationId,
      teamId,
    }: {
      data: { agent_ids: string[]; folder_id: string | null };
      organizationId: string | null;
      teamId: string | null;
    }) =>
      postV2BulkMoveAgents(data, getTenantRequestInit(organizationId, teamId)),
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
  });

  function handleAgentDrop(agentId: string, folderId: string) {
    const agent = agents.find((candidate) => candidate.id === agentId);
    const folder = rawFoldersData?.folders.find(
      (candidate) => candidate.id === folderId,
    );
    if (!agent || !folder) return;
    if (
      (agent.organization_id ?? null) !== (folder.organization_id ?? null) ||
      (agent.team_id ?? null) !== (folder.team_id ?? null)
    ) {
      toast({
        title: "Choose a folder in the same team",
        description:
          "Agents cannot be moved across organization or team folders.",
        variant: "destructive",
      });
      return;
    }
    moveAgentToFolder({
      data: {
        agent_ids: [agentId],
        folder_id: folderId,
      },
      organizationId: agent.organization_id ?? null,
      teamId: agent.team_id ?? null,
    });
  }

  const { data: currentFolderData } = useGetV2GetFolder(
    selectedFolderId ?? "",
    {
      query: {
        select: okData,
        enabled: !!selectedFolderId && isTenantReady,
        queryKey: getTeamScopedQueryKey(
          getGetV2GetFolderQueryKey(selectedFolderId ?? undefined),
          queryOrganizationId,
          queryTeamId,
        ),
      },
      request: getTenantRequestInit(
        queryOrganizationId,
        queryTeamId,
        isTenantReady,
      ),
    },
  );
  const currentFolder = selectedFolderId ? currentFolderData : null;

  const showFolders = !isFavoritesTab;

  const { data: executions } = useGetV1ListAllExecutions({
    query: {
      enabled: isTenantReady,
      queryKey: getTeamScopedQueryKey(
        getGetV1ListAllExecutionsQueryKey(),
        activeOrgID,
        activeTeamID,
      ),
      select: okData,
    },
    request: getTenantRequestInit(activeOrgID, activeTeamID, isTenantReady),
  });

  const { activeGraphIds, errorGraphIds, completedGraphIds } = useMemo(() => {
    const active = new Set<string>();
    const errors = new Set<string>();
    const completed = new Set<string>();
    const cutoff = Date.now() - 72 * 60 * 60 * 1000;
    for (const exec of executions ?? []) {
      const executionKey = getTenantEntityKey(
        exec.graph_id,
        exec.organization_id,
        exec.team_id,
      );
      if (
        exec.status === AgentExecutionStatus.RUNNING ||
        exec.status === AgentExecutionStatus.QUEUED ||
        exec.status === AgentExecutionStatus.REVIEW
      ) {
        active.add(executionKey);
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
        errors.add(executionKey);
      }
      if (exec.status === AgentExecutionStatus.COMPLETED && endedTs > cutoff) {
        completed.add(executionKey);
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

  function handleFolderSelect(folder: LibraryFolder | null) {
    setSelectedFolderScope(
      folder
        ? {
            organizationId: folder.organization_id ?? null,
            teamId: folder.team_id ?? null,
          }
        : null,
    );
    onFolderSelect(folder?.id ?? null);
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
    handleFolderSelect,
    handleFolderDeleted,
  };
}

function filterAgentsByStatus<
  T extends {
    graph_id: string;
    organization_id?: string | null;
    team_id?: string | null;
    has_external_trigger: boolean;
    is_scheduled?: boolean;
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
    const agentKey = getTenantEntityKey(
      agent.graph_id,
      agent.organization_id,
      agent.team_id,
    );
    const isRunning = activeGraphIds.has(agentKey);
    const hasError = errorGraphIds.has(agentKey);
    const isScheduled = isAgentScheduled(agent);

    if (statusFilter === "running") return isRunning;
    if (statusFilter === "attention") return hasError && !isRunning;
    if (statusFilter === "completed") return completedGraphIds.has(agentKey);
    if (statusFilter === "listening")
      return !isRunning && !hasError && agent.has_external_trigger;
    if (statusFilter === "scheduled")
      return (
        !isRunning && !hasError && !agent.has_external_trigger && isScheduled
      );
    if (statusFilter === "idle")
      return (
        !isRunning && !hasError && !agent.has_external_trigger && !isScheduled
      );
    if (statusFilter === "healthy") return !hasError;
    return true;
  });
}
