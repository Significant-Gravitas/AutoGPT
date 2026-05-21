"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  getGetV2GetGraphQueryKey,
  getGetV2GetMemoryOverviewQueryKey,
  useGetV2GetGraph,
  useGetV2GetMemoryOverview,
  usePostV2RebuildCommunities,
  usePostV2TriggerDreamPass,
  usePostV2TriggerNightlyBatch,
  usePostV2TriggerRatificationPass,
} from "@/app/api/__generated__/endpoints/admin/admin";
import type { DreamPassResponse } from "@/app/api/__generated__/models/dreamPassResponse";
import type { GraphResponse } from "@/app/api/__generated__/models/graphResponse";
import type { MemoryOverview } from "@/app/api/__generated__/models/memoryOverview";
import type { NightlyBatchResponse } from "@/app/api/__generated__/models/nightlyBatchResponse";
import type { RatificationResultResponse } from "@/app/api/__generated__/models/ratificationResultResponse";
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

  const dream = usePostV2TriggerDreamPass({
    mutation: {
      onSuccess: (res) => {
        const result = res.data as DreamPassResponse;
        if (result.skipped) {
          toast({
            title: "Dream pass skipped",
            description: `${result.skip_reason ?? "no_reason"} — ${
              result.elapsed_seconds?.toFixed(2) ?? "?"
            }s`,
          });
        } else if (result.error) {
          toast({
            title: "Dream pass failed",
            description: result.error,
            variant: "destructive",
          });
        } else {
          toast({
            title: "Dream pass complete",
            description:
              `${result.elapsed_seconds?.toFixed(1) ?? "?"}s — ` +
              `writes=${result.consolidated_count}, ` +
              `proposals=${result.proposal_count}, ` +
              `demotions=${result.demotion_count}` +
              (result.summary_for_user ? `\n\n${result.summary_for_user}` : ""),
          });
        }
        // A dream writes new memories + may demote edges — both views
        // change. Force a refresh of the overview and the graph.
        queryClient.invalidateQueries({
          queryKey: getGetV2GetMemoryOverviewQueryKey(USER_ID),
        });
        queryClient.invalidateQueries({
          queryKey: getGetV2GetGraphQueryKey(USER_ID),
        });
      },
      onError: (error: Error) => {
        toast({
          title: "Dream pass failed",
          description: error.message,
          variant: "destructive",
        });
      },
    },
  });

  const ratification = usePostV2TriggerRatificationPass({
    mutation: {
      onSuccess: (res) => {
        const result = res.data as RatificationResultResponse;
        if (result.error) {
          toast({
            title: "Ratification failed",
            description: result.error,
            variant: "destructive",
          });
        } else {
          toast({
            title: "Ratification complete",
            description:
              `examined=${result.examined_count}, ` +
              `ratified=${result.ratified_count}, ` +
              `superseded=${result.superseded_count}`,
          });
        }
        queryClient.invalidateQueries({
          queryKey: getGetV2GetMemoryOverviewQueryKey(USER_ID),
        });
        queryClient.invalidateQueries({
          queryKey: getGetV2GetGraphQueryKey(USER_ID),
        });
      },
      onError: (error: Error) => {
        toast({
          title: "Ratification failed",
          description: error.message,
          variant: "destructive",
        });
      },
    },
  });

  const nightly = usePostV2TriggerNightlyBatch({
    mutation: {
      onSuccess: (res) => {
        const result = res.data as NightlyBatchResponse;
        if (result.skipped) {
          toast({
            title: "Nightly batch skipped",
            description: `${result.skip_reason ?? "no_reason"} — ${
              result.elapsed_seconds?.toFixed(2) ?? "?"
            }s`,
          });
        } else if (result.error) {
          toast({
            title: "Nightly batch failed",
            description: result.error,
            variant: "destructive",
          });
        } else {
          const dreamSummary = result.dream
            ? `dream writes=${result.dream.consolidated_count}, ` +
              `proposals=${result.dream.proposal_count}`
            : "dream skipped";
          const ratSummary = result.ratification
            ? `ratified=${result.ratification.ratified_count}, ` +
              `superseded=${result.ratification.superseded_count}`
            : "ratification skipped";
          toast({
            title: "Nightly batch complete",
            description: `${
              result.elapsed_seconds?.toFixed(1) ?? "?"
            }s — ${dreamSummary}; ${ratSummary}`,
          });
        }
        queryClient.invalidateQueries({
          queryKey: getGetV2GetMemoryOverviewQueryKey(USER_ID),
        });
        queryClient.invalidateQueries({
          queryKey: getGetV2GetGraphQueryKey(USER_ID),
        });
      },
      onError: (error: Error) => {
        toast({
          title: "Nightly batch failed",
          description: error.message,
          variant: "destructive",
        });
      },
    },
  });

  function triggerRebuild() {
    rebuild.mutate({ userId: USER_ID, params: { force } });
  }

  function triggerDream() {
    dream.mutate({ userId: USER_ID });
  }

  function triggerRatification() {
    ratification.mutate({ userId: USER_ID });
  }

  function triggerNightly() {
    nightly.mutate({ userId: USER_ID });
  }

  // Narrow the Orval-union response shapes to their success types once
  // here so consumers don't keep re-asserting. react-query's `error`
  // handles the non-200 cases.
  const overviewData = overview.data?.data as MemoryOverview | undefined;
  const graphData = graph.data?.data as GraphResponse | undefined;
  const rebuildData = rebuild.data?.data as RebuildResponse | undefined;
  const dreamData = dream.data?.data as DreamPassResponse | undefined;
  const nightlyData = nightly.data?.data as NightlyBatchResponse | undefined;

  return {
    overview,
    graph,
    rebuild,
    dream,
    ratification,
    nightly,
    triggerRebuild,
    triggerDream,
    triggerRatification,
    triggerNightly,
    force,
    setForce,
    includeEpisodes,
    setIncludeEpisodes,
    includeCommunities,
    setIncludeCommunities,
    overviewData,
    graphData,
    rebuildData,
    dreamData,
    nightlyData,
  };
}
