"use client";

import { useEffect, useMemo } from "react";

import {
  okData,
  getPaginationNextPageNumber,
  getPaginatedTotalCount,
  unpaginate,
} from "@/app/api/helpers";
import { useGetV1ListGraphExecutionsInfinite } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV2ListTriggerAgents } from "@/app/api/__generated__/endpoints/library/library";
import { useGetV2ListPresets } from "@/app/api/__generated__/endpoints/presets/presets";
import { useGetV1ListExecutionSchedulesForAGraph } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useExecutionEvents } from "@/hooks/useExecutionEvents";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryStates } from "nuqs";

function parseTab(
  value: string | null,
): "runs" | "scheduled" | "templates" | "triggers" {
  if (
    value === "runs" ||
    value === "scheduled" ||
    value === "templates" ||
    value === "triggers"
  ) {
    return value;
  }
  return "runs";
}

type Args = {
  graphId: string;
  libraryAgentID: string;
  onSelectRun: (
    runId: string,
    tab?: "runs" | "scheduled" | "templates" | "triggers",
  ) => void;
  onCountsChange?: (info: {
    runsCount: number;
    schedulesCount: number;
    templatesCount: number;
    triggersCount: number;
    loading?: boolean;
  }) => void;
};

export function useSidebarRunsList({
  graphId,
  libraryAgentID,
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
    graphId,
    { page: 1, page_size: 20 },
    {
      query: {
        enabled: !!graphId,
        refetchOnWindowFocus: false,
        getNextPageParam: getPaginationNextPageNumber,
      },
    },
  );

  const schedulesQuery = useGetV1ListExecutionSchedulesForAGraph(graphId, {
    query: {
      enabled: !!graphId,
      select: okData,
    },
  });

  const presetsQuery = useGetV2ListPresets(
    { graph_id: graphId, page: 1, page_size: 100 },
    {
      query: {
        enabled: !!graphId,
        select: (r) => okData(r)?.presets,
      },
    },
  );

  const triggerAgentsEnabled = useGetFlag(Flag.GENERIC_TRIGGER_AGENTS);
  const triggerAgentsQuery = useGetV2ListTriggerAgents(libraryAgentID, {
    query: {
      enabled: triggerAgentsEnabled && !!libraryAgentID,
      select: okData,
    },
  });

  const runs = useMemo(
    () => (runsQuery.data ? unpaginate(runsQuery.data, "executions") : []),
    [runsQuery.data],
  );

  const schedules = schedulesQuery.data || [];
  const allPresets = presetsQuery.data || [];
  const triggers = useMemo(
    () => allPresets.filter((preset) => preset.webhook_id),
    [allPresets],
  );
  const templates = useMemo(
    () => allPresets.filter((preset) => !preset.webhook_id),
    [allPresets],
  );
  const triggerAgents = triggerAgentsEnabled
    ? triggerAgentsQuery.data || []
    : [];

  const runsCount = getPaginatedTotalCount(runsQuery.data, runs.length);
  const schedulesCount = schedules.length;
  const templatesCount = templates.length;
  // Combined count: webhook triggers + trigger agents (both shown
  // under the "Triggers" tab as separate sections).
  const triggersCount = triggers.length + triggerAgents.length;
  const loading =
    !runsQuery.isSuccess ||
    !schedulesQuery.isSuccess ||
    !presetsQuery.isSuccess ||
    (triggerAgentsEnabled && !triggerAgentsQuery.isSuccess);
  const stale =
    runsQuery.isStale ||
    schedulesQuery.isStale ||
    presetsQuery.isStale ||
    (triggerAgentsEnabled && triggerAgentsQuery.isStale);

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
    if (onCountsChange && !stale) {
      onCountsChange({
        runsCount,
        schedulesCount,
        templatesCount,
        triggersCount,
        loading,
      });
    }
  }, [
    onCountsChange,
    runsCount,
    schedulesCount,
    templatesCount,
    triggersCount,
    loading,
    stale,
  ]);

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

  useEffect(() => {
    if (templates.length > 0 && tabValue === "templates" && !activeItem) {
      onSelectRun(templates[0].id, "templates");
    }
  }, [templates, activeItem, tabValue, onSelectRun]);

  useEffect(() => {
    if (tabValue !== "triggers" || activeItem) return;
    if (triggers.length > 0) {
      onSelectRun(triggers[0].id, "triggers");
    } else if (triggerAgents.length > 0) {
      onSelectRun(triggerAgents[0].id, "triggers");
    }
  }, [triggers, triggerAgents, activeItem, tabValue, onSelectRun]);

  return {
    runs,
    schedules,
    templates,
    triggers,
    triggerAgents,
    error:
      schedulesQuery.error ||
      runsQuery.error ||
      presetsQuery.error ||
      (triggerAgentsEnabled ? triggerAgentsQuery.error : null),
    loading,
    runsQuery,
    tabValue,
    runsCount,
    schedulesCount,
    templatesCount,
    triggersCount,
    fetchMoreRuns: runsQuery.fetchNextPage,
    hasMoreRuns: runsQuery.hasNextPage,
    isFetchingMoreRuns: runsQuery.isFetchingNextPage,
  };
}
