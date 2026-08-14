import {
  getListExpertsQueryKey,
  useInstallExpertWorkflow,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { Expert } from "@/app/api/__generated__/models/expert";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { getPaginationNextPageNumber, unpaginate } from "@/app/api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  getAdoptTargetKey,
  getAdoptTargetVersionId,
  getUnadoptedAgents,
  getVisibleGroups,
  WhatRunsFilter,
} from "./helpers";

const AGENTS_PAGE_SIZE = 100;

interface Args {
  experts: Expert[];
  schedules: GraphExecutionJobInfo[];
  enabled: boolean;
}

export function useWhatRunsZone({ experts, schedules, enabled }: Args) {
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState<WhatRunsFilter>("all");
  const [pendingLibraryAgentIds, setPendingLibraryAgentIds] = useState<
    Set<string>
  >(new Set());
  const [adoptedTargetKeys, setAdoptedTargetKeys] = useState<Set<string>>(
    new Set(),
  );

  const agentsQuery = useGetV2ListLibraryAgentsInfinite(
    { page: 1, page_size: AGENTS_PAGE_SIZE, is_hidden: false },
    {
      query: {
        enabled,
        getNextPageParam: getPaginationNextPageNumber,
      },
    },
  );

  const { mutateAsync: installWorkflow } = useInstallExpertWorkflow();

  const libraryAgents = agentsQuery.data
    ? unpaginate(agentsQuery.data, "agents")
    : [];
  const unadoptedAgents = getUnadoptedAgents(
    libraryAgents,
    experts,
    adoptedTargetKeys,
  );
  const groups = getVisibleGroups(experts, schedules, filter);
  const showAgents = filter === "all" || filter === "agents";

  async function adopt(agent: LibraryAgent, expert: Expert) {
    const versionId = getAdoptTargetVersionId(agent);
    if (!versionId || pendingLibraryAgentIds.has(agent.id)) return;
    setPendingLibraryAgentIds((current) => new Set(current).add(agent.id));
    try {
      await installWorkflow({
        expertId: expert.id,
        data: { store_listing_version_id: versionId },
      });
      setAdoptedTargetKeys((current) =>
        new Set(current).add(getAdoptTargetKey(agent, expert)),
      );
      await queryClient.invalidateQueries({
        queryKey: getListExpertsQueryKey(),
      });
      toast({
        title: `Added to ${expert.name}'s workflows`,
        variant: "success",
      });
    } catch (error) {
      const versionUnavailable =
        error instanceof ApiError && error.status === 404;
      toast({
        title: `Couldn't adopt ${agent.name}`,
        description: versionUnavailable
          ? "This Marketplace version is no longer available."
          : "Something went wrong. Please try again.",
        variant: "destructive",
      });
    } finally {
      setPendingLibraryAgentIds((current) => {
        const next = new Set(current);
        next.delete(agent.id);
        return next;
      });
    }
  }

  return {
    filter,
    setFilter,
    groups,
    showAgents,
    unadoptedAgents,
    libraryAgentCount: libraryAgents.length,
    isLoadingAgents: enabled && agentsQuery.isLoading,
    isErrorAgents: agentsQuery.isError && !agentsQuery.data,
    hasMoreAgents: agentsQuery.hasNextPage === true,
    isLoadingMoreAgents: agentsQuery.isFetchingNextPage,
    isErrorLoadingMoreAgents: agentsQuery.isFetchNextPageError,
    retryAgents: () => void agentsQuery.refetch(),
    loadMoreAgents: () => void agentsQuery.fetchNextPage(),
    adopt,
    pendingLibraryAgentIds,
    adoptedTargetKeys,
  };
}
