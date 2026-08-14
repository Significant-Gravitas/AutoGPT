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
import { useEffect, useState } from "react";
import {
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
  const [adoptedGraphIds, setAdoptedGraphIds] = useState<Set<string>>(
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

  useEffect(() => {
    if (
      !enabled ||
      !agentsQuery.hasNextPage ||
      agentsQuery.isFetchingNextPage ||
      agentsQuery.isError
    ) {
      return;
    }
    void agentsQuery.fetchNextPage();
  }, [
    agentsQuery.fetchNextPage,
    agentsQuery.hasNextPage,
    agentsQuery.isError,
    agentsQuery.isFetchingNextPage,
    enabled,
  ]);

  const { mutateAsync: installWorkflow } = useInstallExpertWorkflow();

  const libraryAgents = agentsQuery.data
    ? unpaginate(agentsQuery.data, "agents")
    : [];
  const unadoptedAgents = getUnadoptedAgents(libraryAgents, experts).filter(
    (agent) => !adoptedGraphIds.has(agent.graph_id),
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
      setAdoptedGraphIds((current) => new Set(current).add(agent.graph_id));
      await queryClient.invalidateQueries({
        queryKey: getListExpertsQueryKey(),
      });
      toast({
        title: `Added to ${expert.name}'s workflows`,
        variant: "success",
      });
    } catch {
      toast({
        title: `Couldn't adopt ${agent.name}`,
        description: "Something went wrong. Please try again.",
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
    isLoadingAgents:
      enabled &&
      !agentsQuery.isError &&
      (agentsQuery.isLoading ||
        agentsQuery.isFetchingNextPage ||
        agentsQuery.hasNextPage === true),
    isErrorAgents: agentsQuery.isError,
    retryAgents: () => agentsQuery.refetch(),
    adopt,
    pendingLibraryAgentIds,
  };
}
