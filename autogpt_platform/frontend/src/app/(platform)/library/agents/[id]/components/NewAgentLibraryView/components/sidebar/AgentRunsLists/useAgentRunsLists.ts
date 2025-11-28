"use client";

import { useEffect, useMemo, useState } from "react";

import { useGetV1ListGraphExecutionsInfinite } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV1ListExecutionSchedulesForAGraph } from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { okData } from "@/app/api/helpers";
import { useSearchParams } from "next/navigation";
import {
  computeRunsCount,
  extractRunsFromPages,
  getNextRunsPageParam,
  getRunsPollingInterval,
} from "./helpers";

type Args = {
  graphId?: string;
  onSelectRun: (runId: string) => void;
  onCountsChange?: (info: {
    runsCount: number;
    schedulesCount: number;
    loading?: boolean;
  }) => void;
};

export function useAgentRunsLists({
  graphId,
  onSelectRun,
  onCountsChange,
}: Args) {
  const params = useSearchParams();
  const existingRunId = params.get("executionId") as string | undefined;
  const [tabValue, setTabValue] = useState<"runs" | "scheduled">("runs");

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

  const runs = useMemo(
    () => extractRunsFromPages(runsQuery.data),
    [runsQuery.data],
  );

  const schedules = schedulesQuery.data || [];

  const runsCount = computeRunsCount(runsQuery.data, runs.length);
  const schedulesCount = schedules.length;
  const loading = !schedulesQuery.isSuccess || !runsQuery.isSuccess;

  // Notify parent about counts and loading state
  useEffect(() => {
    if (onCountsChange) {
      onCountsChange({ runsCount, schedulesCount, loading });
    }
  }, [runsCount, schedulesCount, loading, onCountsChange]);

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
    else setTabValue("runs");
  }, [existingRunId]);

  // If there are no runs but there are schedules, and nothing is selected, auto-select the first schedule
  useEffect(() => {
    if (!existingRunId && runs.length === 0 && schedules.length > 0)
      onSelectRun(`schedule:${schedules[0].id}`);
  }, [existingRunId, runs.length, schedules, onSelectRun]);

  return {
    runs,
    schedules,
    error: schedulesQuery.error || runsQuery.error,
    loading,
    runsQuery,
    tabValue,
    setTabValue,
    runsCount,
    schedulesCount,
    fetchMoreRuns: runsQuery.fetchNextPage,
    hasMoreRuns: runsQuery.hasNextPage,
    isFetchingMoreRuns: runsQuery.isFetchingNextPage,
  };
}
