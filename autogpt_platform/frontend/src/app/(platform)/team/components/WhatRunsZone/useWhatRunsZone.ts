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
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  getAdoptTargetKey,
  getAdoptTargetVersionID,
  getUnadoptedAgents,
  getVisibleGroups,
  pruneAdoptedTargetKeys,
  WhatRunsFilter,
} from "./helpers";

const AGENTS_PAGE_SIZE = 100;

function hasStatus(error: unknown, status: number) {
  return (
    typeof error === "object" &&
    error !== null &&
    "status" in error &&
    error.status === status
  );
}

interface Args {
  experts: Expert[];
  schedules: GraphExecutionJobInfo[];
  enabled: boolean;
}

export function useWhatRunsZone({ experts, schedules, enabled }: Args) {
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState<WhatRunsFilter>("all");
  const [pendingLibraryAgentIDs, setPendingLibraryAgentIDs] = useState<
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
    const versionID = getAdoptTargetVersionID(agent);
    if (!versionID || pendingLibraryAgentIDs.has(agent.id)) return;
    const targetKey = getAdoptTargetKey(agent, expert);
    setPendingLibraryAgentIDs((current) => new Set(current).add(agent.id));
    try {
      await installWorkflow({
        expertId: expert.id,
        data: { store_listing_version_id: versionID },
      });
      setAdoptedTargetKeys((current) => {
        const next = new Set(
          pruneAdoptedTargetKeys(current, libraryAgents, experts),
        );
        next.add(targetKey);
        return next;
      });
      await queryClient.invalidateQueries({
        queryKey: getListExpertsQueryKey(),
      });
      toast({
        title: `Added to ${expert.name}'s workflows`,
        variant: "success",
      });
    } catch (error) {
      const versionUnavailable = hasStatus(error, 404);
      toast({
        title: `Couldn't adopt ${agent.name}`,
        description: versionUnavailable
          ? "This Marketplace version is no longer available."
          : "Something went wrong. Please try again.",
        variant: "destructive",
      });
    } finally {
      setPendingLibraryAgentIDs((current) => {
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
    pendingLibraryAgentIDs,
    adoptedTargetKeys,
  };
}
