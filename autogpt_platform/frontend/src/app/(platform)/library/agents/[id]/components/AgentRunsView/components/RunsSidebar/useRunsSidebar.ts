"use client";

import { useEffect, useMemo, useState } from "react";

import { useGetV1ListGraphExecutionsInfinite } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV1ListExecutionSchedulesForAGraph } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useGetV2ListPresetsInfinite } from "@/app/api/__generated__/endpoints/presets/presets";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { useSearchParams } from "next/navigation";
import { okData } from "@/app/api/helpers";
import {
  getPaginatedTotalCount,
  getPaginationNextPageNumber,
  getRunsPollingInterval,
  unpaginate,
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
        getNextPageParam: getPaginationNextPageNumber,
      },
    },
  );

  const presetsQuery = useGetV2ListPresetsInfinite(
    { graph_id: graphId || null, page: 1, page_size: 100 },
    {
      query: {
        enabled: !!graphId,
        refetchOnWindowFocus: false,
        getNextPageParam: getPaginationNextPageNumber,
      },
    },
  );

  const schedulesQuery = useGetV1ListExecutionSchedulesForAGraph(
    graphId || "",
    {
      query: {
        enabled: !!graphId,
        select: (r) => okData(r) ?? [],
      },
    },
  );

  const runs = useMemo(
    () => (runsQuery.data ? unpaginate(runsQuery.data, "executions") : []),
    [runsQuery.data],
  );
  const presets = useMemo(
    () => (presetsQuery.data ? unpaginate(presetsQuery.data, "presets") : []),
    [presetsQuery.data],
  );
  const schedules = schedulesQuery.data || [];

  const runsCount = getPaginatedTotalCount(runsQuery.data, runs.length);
  const presetsCount = getPaginatedTotalCount(
    presetsQuery.data,
    presets.length,
  );
  const schedulesCount = schedules.length;
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
    presets,
    schedules,
    error: schedulesQuery.error || runsQuery.error || presetsQuery.error,
    loading,
    runsQuery,
    presetsQuery,
    tabValue,
    setTabValue,
    runsCount,
    presetsCount,
    schedulesCount,
    hasMoreRuns: runsQuery.hasNextPage,
    fetchMoreRuns: runsQuery.fetchNextPage,
    isFetchingMoreRuns: runsQuery.isFetchingNextPage,
    hasMorePresets: presetsQuery.hasNextPage,
    fetchMorePresets: presetsQuery.fetchNextPage,
    isFetchingMorePresets: presetsQuery.isFetchingNextPage,
  };
}
