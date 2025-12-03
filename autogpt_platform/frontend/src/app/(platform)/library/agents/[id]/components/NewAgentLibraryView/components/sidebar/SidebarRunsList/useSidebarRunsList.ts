"use client";

import { useEffect, useMemo } from "react";

import { useGetV1ListGraphExecutionsInfinite } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV1ListExecutionSchedulesForAGraph } from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { okData } from "@/app/api/helpers";
import { useExecutionEvents } from "@/hooks/useExecutionEvents";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryStates } from "nuqs";
import {
  computeRunsCount,
  extractRunsFromPages,
  getNextRunsPageParam,
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
  const queryClient = useQueryClient();

  const runsQuery = useGetV1ListGraphExecutionsInfinite(
    graphId || "",
    { page: 1, page_size: 20 },
    {
      query: {
        enabled: !!graphId,
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

  // Update query cache when execution events arrive via websocket
  useExecutionEvents({
    graphId: graphId || undefined,
    enabled: !!graphId && tabValue === "runs",
    onExecutionUpdate: (_execution) => {
      // Invalidate and refetch the query to ensure we have the latest data
      // This is simpler and more reliable than manually updating the cache
      // The queryKey is stable and includes the graphId, so this only invalidates
      // queries for this specific graph's executions
      queryClient.invalidateQueries({ queryKey: runsQuery.queryKey });
    },
  });

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
