"use client";

import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  getGetV2GetCommunityRebuildStatusQueryKey,
  getGetV2GetDreamPassStatusQueryKey,
  getGetV2GetGraphQueryKey,
  getGetV2GetMemoryOverviewQueryKey,
  getGetV2GetNightlyBatchStatusQueryKey,
  useGetV2GetCommunityRebuildStatus,
  useGetV2GetDreamPassStatus,
  useGetV2GetGraph,
  useGetV2GetMemoryOverview,
  useGetV2GetNightlyBatchStatus,
  usePostV2RebuildCommunities,
  usePostV2TriggerDreamPass,
  usePostV2TriggerNightlyBatch,
  usePostV2TriggerRatificationPass,
} from "@/app/api/__generated__/endpoints/admin/admin";
import type { GraphResponse } from "@/app/api/__generated__/models/graphResponse";
import type { JobStatusResponse } from "@/app/api/__generated__/models/jobStatusResponse";
import type { JobStatusResponseState } from "@/app/api/__generated__/models/jobStatusResponseState";
import type { JobTriggerResponse } from "@/app/api/__generated__/models/jobTriggerResponse";
import type { MemoryOverview } from "@/app/api/__generated__/models/memoryOverview";
import type { RatificationResultResponse } from "@/app/api/__generated__/models/ratificationResultResponse";
import { useToast } from "@/components/molecules/Toast/use-toast";

const USER_ID = "me";

// Polling cadence while a job is in flight. 3s is fast enough that the
// admin sees phase transitions promptly without flooding the backend
// (the BatchExecutor itself only walks every 10s anyway).
const POLL_INTERVAL_MS = 3000;

type TerminalState = Extract<JobStatusResponseState, "complete" | "errored">;

function isTerminal(
  state: JobStatusResponseState | undefined,
): state is TerminalState {
  return state === "complete" || state === "errored";
}

export function useMemoryVisualizer() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [force, setForce] = useState(false);
  const [includeEpisodes, setIncludeEpisodes] = useState(false);
  const [includeCommunities, setIncludeCommunities] = useState(true);

  // Track the in-flight job per kind so the polling hooks know what
  // to watch. ``undefined`` means no job in flight for that kind.
  const [activeDreamJobId, setActiveDreamJobId] = useState<
    string | undefined
  >();
  const [activeNightlyJobId, setActiveNightlyJobId] = useState<
    string | undefined
  >();
  const [activeRebuildJobId, setActiveRebuildJobId] = useState<
    string | undefined
  >();

  const overview = useGetV2GetMemoryOverview(USER_ID);
  const graph = useGetV2GetGraph(USER_ID, {
    include_episodes: includeEpisodes,
    include_communities: includeCommunities,
    node_limit: 10000,
    edge_limit: 20000,
  });

  // --- Triggers (POST → 202 + job_id) ---------------------------------------

  const rebuild = usePostV2RebuildCommunities({
    mutation: {
      onSuccess: (res) => {
        const result = res.data as JobTriggerResponse;
        setActiveRebuildJobId(result.job_id);
      },
      onError: (error: Error) => {
        toast({
          title: "Rebuild failed to schedule",
          description: error.message,
          variant: "destructive",
        });
      },
    },
  });

  const dream = usePostV2TriggerDreamPass({
    mutation: {
      onSuccess: (res) => {
        const result = res.data as JobTriggerResponse;
        setActiveDreamJobId(result.job_id);
      },
      onError: (error: Error) => {
        toast({
          title: "Dream pass failed to schedule",
          description: error.message,
          variant: "destructive",
        });
      },
    },
  });

  const nightly = usePostV2TriggerNightlyBatch({
    mutation: {
      onSuccess: (res) => {
        const result = res.data as JobTriggerResponse;
        setActiveNightlyJobId(result.job_id);
      },
      onError: (error: Error) => {
        toast({
          title: "Nightly batch failed to schedule",
          description: error.message,
          variant: "destructive",
        });
      },
    },
  });

  // Ratification stays synchronous — Cypher-only, finishes in seconds,
  // no JobStatus row to poll for.
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

  // --- Status pollers (GET → JobStatusResponse) ----------------------------

  const dreamStatus = useGetV2GetDreamPassStatus(
    USER_ID,
    activeDreamJobId ?? "",
    {
      query: {
        enabled: !!activeDreamJobId,
        refetchInterval: (query) =>
          isTerminal(
            (query.state.data?.data as JobStatusResponse | undefined)?.state,
          )
            ? false
            : POLL_INTERVAL_MS,
        staleTime: 0,
      },
    },
  );

  const nightlyStatus = useGetV2GetNightlyBatchStatus(
    USER_ID,
    activeNightlyJobId ?? "",
    {
      query: {
        enabled: !!activeNightlyJobId,
        refetchInterval: (query) =>
          isTerminal(
            (query.state.data?.data as JobStatusResponse | undefined)?.state,
          )
            ? false
            : POLL_INTERVAL_MS,
        staleTime: 0,
      },
    },
  );

  const rebuildStatus = useGetV2GetCommunityRebuildStatus(
    USER_ID,
    activeRebuildJobId ?? "",
    {
      query: {
        enabled: !!activeRebuildJobId,
        refetchInterval: (query) =>
          isTerminal(
            (query.state.data?.data as JobStatusResponse | undefined)?.state,
          )
            ? false
            : POLL_INTERVAL_MS,
        staleTime: 0,
      },
    },
  );

  // --- Terminal-state handlers --------------------------------------------
  //
  // When a job hits ``complete`` or ``errored``: toast, invalidate the
  // memory cache (so the visualizer refreshes with the new writes /
  // demotions), clear the active id (stops polling).

  useTerminalEffect(
    "Dream pass",
    activeDreamJobId,
    dreamStatus.data?.data as JobStatusResponse | undefined,
    () => {
      setActiveDreamJobId(undefined);
      queryClient.invalidateQueries({
        queryKey: getGetV2GetDreamPassStatusQueryKey(USER_ID, activeDreamJobId),
      });
      queryClient.invalidateQueries({
        queryKey: getGetV2GetMemoryOverviewQueryKey(USER_ID),
      });
      queryClient.invalidateQueries({
        queryKey: getGetV2GetGraphQueryKey(USER_ID),
      });
    },
    toast,
  );

  useTerminalEffect(
    "Nightly batch",
    activeNightlyJobId,
    nightlyStatus.data?.data as JobStatusResponse | undefined,
    () => {
      setActiveNightlyJobId(undefined);
      queryClient.invalidateQueries({
        queryKey: getGetV2GetNightlyBatchStatusQueryKey(
          USER_ID,
          activeNightlyJobId,
        ),
      });
      queryClient.invalidateQueries({
        queryKey: getGetV2GetMemoryOverviewQueryKey(USER_ID),
      });
      queryClient.invalidateQueries({
        queryKey: getGetV2GetGraphQueryKey(USER_ID),
      });
    },
    toast,
  );

  useTerminalEffect(
    "Community rebuild",
    activeRebuildJobId,
    rebuildStatus.data?.data as JobStatusResponse | undefined,
    () => {
      setActiveRebuildJobId(undefined);
      queryClient.invalidateQueries({
        queryKey: getGetV2GetCommunityRebuildStatusQueryKey(
          USER_ID,
          activeRebuildJobId,
        ),
      });
      queryClient.invalidateQueries({
        queryKey: getGetV2GetMemoryOverviewQueryKey(USER_ID),
      });
      queryClient.invalidateQueries({
        queryKey: getGetV2GetGraphQueryKey(USER_ID),
      });
    },
    toast,
  );

  // --- Action callbacks ---------------------------------------------------

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

  const overviewData = overview.data?.data as MemoryOverview | undefined;
  const graphData = graph.data?.data as GraphResponse | undefined;
  const dreamStatusData = dreamStatus.data?.data as
    | JobStatusResponse
    | undefined;
  const nightlyStatusData = nightlyStatus.data?.data as
    | JobStatusResponse
    | undefined;
  const rebuildStatusData = rebuildStatus.data?.data as
    | JobStatusResponse
    | undefined;

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
    // Polling status + active flags consumed by MemoryVisualizer to
    // render phase-aware button text.
    dreamStatusData,
    nightlyStatusData,
    rebuildStatusData,
    activeDreamJobId,
    activeNightlyJobId,
    activeRebuildJobId,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function useTerminalEffect(
  label: string,
  activeJobId: string | undefined,
  status: JobStatusResponse | undefined,
  onTerminal: () => void,
  toast: ReturnType<typeof useToast>["toast"],
) {
  // Reacts to ``status`` transitioning to a terminal state. Only fires
  // when we have an active job — protects against late polls from a
  // prior run.
  useEffect(() => {
    if (!activeJobId || !status) return;
    if (!isTerminal(status.state)) return;

    if (status.state === "complete") {
      toast({
        title: `${label} complete`,
        description: completionSummary(status),
      });
    } else {
      toast({
        title: `${label} failed`,
        description: status.error ?? "unknown error",
        variant: "destructive",
      });
    }
    onTerminal();
    // ``onTerminal`` and ``toast`` are intentionally omitted — they
    // change on every render and would cause infinite re-runs. The
    // job_id + state transition is what we want to react to.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeJobId, status?.state]);
}

function completionSummary(status: JobStatusResponse): string {
  const result = status.result as Record<string, unknown> | null | undefined;
  if (!result) return "OK";
  // The result shape varies by kind; pick a few common fields and
  // render whatever's present.
  const parts: string[] = [];
  if (typeof result.consolidated_count === "number") {
    parts.push(`writes=${result.consolidated_count}`);
  }
  if (typeof result.proposal_count === "number") {
    parts.push(`proposals=${result.proposal_count}`);
  }
  if (typeof result.demotion_count === "number") {
    parts.push(`demotions=${result.demotion_count}`);
  }
  if (
    typeof result.communities_built === "object" &&
    result.communities_built
  ) {
    parts.push(`communities=${JSON.stringify(result.communities_built)}`);
  }
  if (typeof (result.elapsed_seconds as number | undefined) === "number") {
    parts.push(`${(result.elapsed_seconds as number).toFixed(1)}s`);
  }
  return parts.length ? parts.join(" — ") : "OK";
}
