import {
  getListExpertsQueryKey,
  useInstallExpertWorkflow,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV2ListLibraryAgents } from "@/app/api/__generated__/endpoints/library/library";
import { getV2GetSpecificAgent } from "@/app/api/__generated__/endpoints/store/store";
import { Expert } from "@/app/api/__generated__/models/expert";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  getAdoptableListing,
  getUnadoptedAgents,
  getVisibleGroups,
  WhatRunsFilter,
} from "./helpers";

interface Args {
  experts: Expert[];
  enabled: boolean;
}

export function useWhatRunsZone({ experts, enabled }: Args) {
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState<WhatRunsFilter>("all");
  const [pendingAgentId, setPendingAgentId] = useState<string | null>(null);

  const agentsQuery = useGetV2ListLibraryAgents(
    { page: 1, page_size: 100 },
    { query: { select: okData, enabled } },
  );

  const { mutateAsync: installWorkflow } = useInstallExpertWorkflow();

  const libraryAgents = agentsQuery.data?.agents ?? [];
  const unadoptedAgents = getUnadoptedAgents(libraryAgents, experts);
  const groups = getVisibleGroups(experts, filter);
  const showAgents =
    filter === "agents" || (filter === "all" && unadoptedAgents.length > 0);

  async function adopt(agent: LibraryAgent, expert: Expert) {
    const listing = getAdoptableListing(agent);
    if (!listing) return;
    setPendingAgentId(agent.graph_id);
    try {
      const details = await getV2GetSpecificAgent(
        listing.creator.slug,
        listing.slug,
      );
      if (details.status !== 200) {
        throw new Error("Failed to resolve agent listing");
      }
      await installWorkflow({
        expertId: expert.id,
        data: {
          store_listing_version_id: details.data.store_listing_version_id,
        },
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
      setPendingAgentId(null);
    }
  }

  return {
    filter,
    setFilter,
    groups,
    showAgents,
    unadoptedAgents,
    isLoadingAgents: enabled && agentsQuery.isLoading,
    adopt,
    pendingAgentId,
  };
}
