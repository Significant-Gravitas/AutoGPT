"use client";

import { useEffect, useMemo } from "react";

import { useGetV1ListGraphExecutionsInfinite } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV1ListExecutionSchedulesForAGraph } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useGetV2ListPresetsInfinite } from "@/app/api/__generated__/endpoints/presets/presets";
import { okData } from "@/app/api/helpers";
import { useExecutionEvents } from "@/hooks/useExecutionEvents";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryStates } from "nuqs";
import {
  getPaginatedTotalCount,
  getPaginationNextPageNumber,
  unpaginate,
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
    presetsCount: number;
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
      onCountsChange({ runsCount, schedulesCount, presetsCount, loading });
    }
  }, [runsCount, schedulesCount, presetsCount, loading, onCountsChange]);

  useEffect(() => {
    if (runs.length > 0 && tabValue === "runs" && !activeItem) {
      onSelectRun(runs[0].id, "runs");
    }
  }, [runs, activeItem, tabValue, onSelectRun]);

  // If there are no runs but there are schedules or presets, auto-select the first available
  useEffect(() => {
    if (!activeItem && runs.length === 0) {
      if (schedules.length > 0) {
        onSelectRun(`schedule:${schedules[0].id}`);
      } else if (presets.length > 0) {
        onSelectRun(`preset:${presets[0].id}`);
      }
    }
  }, [activeItem, runs.length, schedules, onSelectRun]);

  return {
    runs,
    presets,
    schedules,
    error: schedulesQuery.error || runsQuery.error || presetsQuery.error,
    loading,
    runsQuery,
    presetsQuery,
    tabValue,
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
