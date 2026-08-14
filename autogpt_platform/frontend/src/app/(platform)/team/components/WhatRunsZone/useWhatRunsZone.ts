import {
  getListExpertsQueryKey,
  useInstallExpertWorkflow,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV2ListLibraryAgents } from "@/app/api/__generated__/endpoints/library/library";
import { Expert } from "@/app/api/__generated__/models/expert";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  getAdoptTargetVersionId,
  getUnadoptedAgents,
  getVisibleGroups,
  WhatRunsFilter,
} from "./helpers";

const AGENTS_PAGE_SIZE = 100;

interface Args {
  experts: Expert[];
  enabled: boolean;
}

export function useWhatRunsZone({ experts, enabled }: Args) {
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState<WhatRunsFilter>("all");
  const [pendingAgentIds, setPendingAgentIds] = useState<Set<string>>(
    new Set(),
  );

  const agentsQuery = useGetV2ListLibraryAgents(
    { page: 1, page_size: AGENTS_PAGE_SIZE },
    { query: { select: okData, enabled } },
  );

  const { mutateAsync: installWorkflow } = useInstallExpertWorkflow();

  const libraryAgents = agentsQuery.data?.agents ?? [];
  const totalAgents =
    agentsQuery.data?.pagination.total_items ?? libraryAgents.length;
  const unadoptedAgents = getUnadoptedAgents(libraryAgents, experts);
  const groups = getVisibleGroups(experts, filter);
  const showAgents = filter === "all" || filter === "agents";

  async function adopt(agent: LibraryAgent, expert: Expert) {
    const versionId = getAdoptTargetVersionId(agent);
    if (!versionId || pendingAgentIds.has(agent.graph_id)) return;
    setPendingAgentIds((current) => new Set(current).add(agent.graph_id));
    try {
      await installWorkflow({
        expertId: expert.id,
        data: { store_listing_version_id: versionId },
      });
      await queryClient.invalidateQueries({
        queryKey: getListExpertsQueryKey(),
      });
      toast({
        title: `Adopted into ${expert.name}'s thread`,
        variant: "success",
      });
    } catch {
      toast({
        title: `Couldn't adopt ${agent.name}`,
        description: "Something went wrong. Please try again.",
        variant: "destructive",
      });
    } finally {
      setPendingAgentIds((current) => {
        const next = new Set(current);
        next.delete(agent.graph_id);
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
    totalAgents,
    hiddenAgentCount: Math.max(0, totalAgents - libraryAgents.length),
    isLoadingAgents: enabled && agentsQuery.isLoading,
    isErrorAgents: agentsQuery.isError,
    retryAgents: () => agentsQuery.refetch(),
    adopt,
    pendingAgentIds,
  };
}
