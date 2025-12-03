"use client";

import { useEffect, useMemo } from "react";

import { useGetV1ListGraphExecutionsInfinite } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV1ListExecutionSchedulesForAGraph } from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { okData } from "@/app/api/helpers";
import { parseAsString, useQueryStates } from "nuqs";
import {
  computeRunsCount,
  extractRunsFromPages,
  getNextRunsPageParam,
  getRunsPollingInterval,
} from "./helpers";

function parseTab(value: string | null): "runs" | "scheduled" | "templates" {
  if (value === "runs" || value === "scheduled" || value === "templates") {
    return value;
  }
  return "runs";
}

type Args = {
  graphId?: string;
  onSelectRun: (runId: string, tab?: "runs" | "scheduled") => void;
  onCountsChange?: (info: {
    runsCount: number;
    schedulesCount: number;
    loading?: boolean;
  }) => void;
};

export function useSidebarRunsList({
  graphId,
  onSelectRun,
  onCountsChange,
}: Args) {
  const [{ activeItem, activeTab: activeTabRaw }] = useQueryStates({
    activeItem: parseAsString,
    activeTab: parseAsString,
  });

  const tabValue = useMemo(() => parseTab(activeTabRaw), [activeTabRaw]);

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
    if (runs.length > 0 && tabValue === "runs" && !activeItem) {
      onSelectRun(runs[0].id, "runs");
    }
  }, [runs, activeItem, tabValue, onSelectRun]);

  // If there are no runs but there are schedules, and nothing is selected, auto-select the first schedule
  useEffect(() => {
    if (!activeItem && runs.length === 0 && schedules.length > 0) {
      onSelectRun(schedules[0].id, "scheduled");
    }
  }, [activeItem, runs.length, schedules, onSelectRun]);

  return {
    runs,
    schedules,
    error: schedulesQuery.error || runsQuery.error,
    loading,
    runsQuery,
    tabValue,
    runsCount,
    schedulesCount,
    fetchMoreRuns: runsQuery.fetchNextPage,
    hasMoreRuns: runsQuery.hasNextPage,
    isFetchingMoreRuns: runsQuery.isFetchingNextPage,
  };
}
