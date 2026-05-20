"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  getGetV2GetMemoryOverviewQueryKey,
  getGetV2ListCommunitiesQueryKey,
  getGetV2ListEntitiesQueryKey,
  getGetV2ListFactsQueryKey,
  useGetV2GetMemoryOverview,
  useGetV2ListCommunities,
  useGetV2ListEntities,
  useGetV2ListFacts,
  usePostV2RebuildCommunities,
} from "@/app/api/__generated__/endpoints/admin/admin";
import type { CommunityListResponse } from "@/app/api/__generated__/models/communityListResponse";
import type { EntityListResponse } from "@/app/api/__generated__/models/entityListResponse";
import type { FactListResponse } from "@/app/api/__generated__/models/factListResponse";
import type { MemoryOverview } from "@/app/api/__generated__/models/memoryOverview";
import type { RebuildResponse } from "@/app/api/__generated__/models/rebuildResponse";
import { useToast } from "@/components/molecules/Toast/use-toast";

const USER_ID = "me";

type StatusFilter = "any" | "active" | "superseded" | "contradicted";

export function useMemoryVisualizer() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("any");
  const [force, setForce] = useState(false);

  const overview = useGetV2GetMemoryOverview(USER_ID);
  const entities = useGetV2ListEntities(USER_ID, { limit: 100 });
  const facts = useGetV2ListFacts(USER_ID, {
    limit: 100,
    status: statusFilter,
  });
  const communities = useGetV2ListCommunities(USER_ID, { limit: 50 });

  const rebuild = usePostV2RebuildCommunities({
    mutation: {
      onSuccess: (res) => {
        // Generated success-or-error union — admin gating means errors
        // surface via react-query's `error`, so on a 200 the data is
        // guaranteed to be the success payload.
        const result = res.data as RebuildResponse;
        if (result.skipped) {
          toast({
            title: "Rebuild skipped",
            description: `${result.skip_reason ?? "no_reason"} — ${
              result.elapsed_seconds?.toFixed(2) ?? "?"
            }s`,
          });
        } else if (result.error) {
          toast({
            title: "Rebuild failed",
            description: result.error,
            variant: "destructive",
          });
        } else {
          toast({
            title: "Rebuild complete",
            description: `${
              result.elapsed_seconds?.toFixed(1) ?? "?"
            }s — ${JSON.stringify(result.communities_built)}`,
          });
        }
        // Refresh all memory views after a real rebuild — a skip
        // doesn't change graph state so we could skip these, but
        // invalidating is cheap and keeps the UI in lockstep.
        queryClient.invalidateQueries({
          queryKey: getGetV2GetMemoryOverviewQueryKey(USER_ID),
        });
        queryClient.invalidateQueries({
          queryKey: getGetV2ListCommunitiesQueryKey(USER_ID),
        });
        queryClient.invalidateQueries({
          queryKey: getGetV2ListEntitiesQueryKey(USER_ID),
        });
        queryClient.invalidateQueries({
          queryKey: getGetV2ListFactsQueryKey(USER_ID),
        });
      },
      onError: (error: Error) => {
        toast({
          title: "Rebuild failed",
          description: error.message,
          variant: "destructive",
        });
      },
    },
  });

  function triggerRebuild() {
    rebuild.mutate({ userId: USER_ID, params: { force } });
  }

  // Narrow the Orval-union response shapes to their success types once
  // here so the consuming component doesn't keep re-asserting.
  const overviewData = overview.data?.data as MemoryOverview | undefined;
  const entitiesData = entities.data?.data as EntityListResponse | undefined;
  const factsData = facts.data?.data as FactListResponse | undefined;
  const communitiesData = communities.data?.data as
    | CommunityListResponse
    | undefined;
  const rebuildData = rebuild.data?.data as RebuildResponse | undefined;

  return {
    overview,
    entities,
    facts,
    communities,
    rebuild,
    triggerRebuild,
    statusFilter,
    setStatusFilter,
    force,
    setForce,
    overviewData,
    entitiesData,
    factsData,
    communitiesData,
    rebuildData,
  };
}
