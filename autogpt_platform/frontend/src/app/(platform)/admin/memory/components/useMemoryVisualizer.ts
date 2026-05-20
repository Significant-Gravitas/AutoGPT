"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  getGetV2GetGraphQueryKey,
  getGetV2GetMemoryOverviewQueryKey,
  useGetV2GetGraph,
  useGetV2GetMemoryOverview,
  usePostV2RebuildCommunities,
} from "@/app/api/__generated__/endpoints/admin/admin";
import type { GraphResponse } from "@/app/api/__generated__/models/graphResponse";
import type { MemoryOverview } from "@/app/api/__generated__/models/memoryOverview";
import type { RebuildResponse } from "@/app/api/__generated__/models/rebuildResponse";
import { useToast } from "@/components/molecules/Toast/use-toast";

const USER_ID = "me";

export function useMemoryVisualizer() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [force, setForce] = useState(false);
  const [includeEpisodes, setIncludeEpisodes] = useState(false);
  const [includeCommunities, setIncludeCommunities] = useState(true);

  const overview = useGetV2GetMemoryOverview(USER_ID);
  const graph = useGetV2GetGraph(USER_ID, {
    include_episodes: includeEpisodes,
    include_communities: includeCommunities,
    node_limit: 10000,
    edge_limit: 20000,
  });

  const rebuild = usePostV2RebuildCommunities({
    mutation: {
      onSuccess: (res) => {
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
        // After a real rebuild the graph and counts both change.
        queryClient.invalidateQueries({
          queryKey: getGetV2GetMemoryOverviewQueryKey(USER_ID),
        });
        queryClient.invalidateQueries({
          queryKey: getGetV2GetGraphQueryKey(USER_ID),
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
  // here so consumers don't keep re-asserting. react-query's `error`
  // handles the non-200 cases.
  const overviewData = overview.data?.data as MemoryOverview | undefined;
  const graphData = graph.data?.data as GraphResponse | undefined;
  const rebuildData = rebuild.data?.data as RebuildResponse | undefined;

  return {
    overview,
    graph,
    rebuild,
    triggerRebuild,
    force,
    setForce,
    includeEpisodes,
    setIncludeEpisodes,
    includeCommunities,
    setIncludeCommunities,
    overviewData,
    graphData,
    rebuildData,
  };
}
