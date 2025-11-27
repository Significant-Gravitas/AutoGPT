"use client";

import { useEffect, useMemo, useState } from "react";

import { useGetV1ListGraphExecutionsInfinite } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV1ListExecutionSchedulesForAGraph } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useGetV2ListPresets } from "@/app/api/__generated__/endpoints/presets/presets";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import type { LibraryAgentPresetResponse } from "@/app/api/__generated__/models/libraryAgentPresetResponse";
import { useSearchParams } from "next/navigation";
import { okData } from "@/app/api/helpers";
import {
  getRunsPollingInterval,
  computeRunsCount,
  getNextRunsPageParam,
  extractRunsFromPages,
} from "./helpers";

type Args = {
  graphId?: string;
  onSelectRun: (runId: string) => void;
  onCountsChange?: (info: {
    runsCount: number;
    schedulesCount: number;
    presetsCount: number;
    loading?: boolean;
  }) => void;
};

export function useRunsSidebar({ graphId, onSelectRun, onCountsChange }: Args) {
  const params = useSearchParams();
  const existingRunId = params.get("executionId") as string | undefined;
  const [tabValue, setTabValue] = useState<"runs" | "scheduled" | "templates">(
    "runs",
  );

  const runsQuery = useGetV1ListGraphExecutionsInfinite(
    graphId || "",
    { page: 1, page_size: 20 },
    {
      query: {
        enabled: !!graphId,
        refetchInterval: (q) =>
          getRunsPollingInterval(q.state.data?.pages, tabValue === "runs"),
        refetchIntervalInBackground: true,
        refetchOnWindowFocus: false,
        getNextPageParam: getNextRunsPageParam,
      },
    },
  );

  const schedulesQuery = useGetV1ListExecutionSchedulesForAGraph(
    graphId || "",
    {
      query: {
        enabled: !!graphId,
        select: (r) => okData<GraphExecutionJobInfo[]>(r) ?? [],
      },
    },
  );

  const presetsQuery = useGetV2ListPresets(
    { graph_id: graphId || null, page: 1, page_size: 100 },
    {
      query: {
        enabled: !!graphId,
        select: (r) => okData<LibraryAgentPresetResponse>(r)?.presets ?? [],
      },
    },
  );

  const runs = useMemo(
    () => extractRunsFromPages(runsQuery.data),
    [runsQuery.data],
  );

  const schedules = schedulesQuery.data || [];
  const presets = presetsQuery.data || [];

  const runsCount = computeRunsCount(runsQuery.data, runs.length);
  const schedulesCount = schedules.length;
  const presetsCount = presets.length;
  const triggersCount = presets.filter((preset) => !!preset.webhook).length;
  const regularPresetsCount = presets.filter(
    (preset) => !preset.webhook,
  ).length;
  const loading =
    !schedulesQuery.isSuccess ||
    !runsQuery.isSuccess ||
    !presetsQuery.isSuccess;

  // Notify parent about counts and loading state
  useEffect(() => {
    if (onCountsChange) {
      onCountsChange({ runsCount, schedulesCount, presetsCount, loading });
    }
  }, [runsCount, schedulesCount, presetsCount, loading, onCountsChange]);

  useEffect(() => {
    if (runs.length > 0) {
      if (existingRunId) {
        onSelectRun(existingRunId);
        return;
      }
      onSelectRun(runs[0].id);
    }
  }, [runs, existingRunId]);

  useEffect(() => {
    if (existingRunId && existingRunId.startsWith("schedule:"))
      setTabValue("scheduled");
    else if (existingRunId && existingRunId.startsWith("preset:"))
      setTabValue("templates");
    else setTabValue("runs");
  }, [existingRunId]);

  // If there are no runs but there are schedules or presets, auto-select the first available
  useEffect(() => {
    if (!existingRunId && runs.length === 0) {
      if (schedules.length > 0) {
        onSelectRun(`schedule:${schedules[0].id}`);
      } else if (presets.length > 0) {
        onSelectRun(`preset:${presets[0].id}`);
      }
    }
  }, [existingRunId, runs.length, schedules, presets, onSelectRun]);

  return {
    runs,
    schedules,
    presets,
    error: schedulesQuery.error || runsQuery.error || presetsQuery.error,
    loading,
    runsQuery,
    presetsQuery,
    tabValue,
    setTabValue,
    runsCount,
    schedulesCount,
    presetsCount,
    triggersCount,
    regularPresetsCount,
    fetchMoreRuns: runsQuery.fetchNextPage,
    hasMoreRuns: runsQuery.hasNextPage,
    isFetchingMoreRuns: runsQuery.isFetchingNextPage,
  };
}
