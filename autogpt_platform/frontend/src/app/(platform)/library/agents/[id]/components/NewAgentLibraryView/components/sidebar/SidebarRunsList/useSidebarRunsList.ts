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
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useGetV1ListExecutionSchedulesForAGraph } from "@/app/api/__generated__/endpoints/schedules/schedules";
import {
  activeItemParamFor,
  isWebhookPreset,
  retryUnlessClientError,
} from "../../../helpers";
import { useAgentPresetsQuery } from "../../../hooks/useAgentPresetsQuery";
import {
  EXECUTION_REFETCH_THROTTLE_MS,
  useExecutionEvents,
} from "@/hooks/useExecutionEvents";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useSchedulerOutage } from "@/services/schedules/useSchedulerOutage";
import { useSchedulerQueryOptions } from "@/services/schedules/useSchedulerQueryOptions";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryStates } from "nuqs";
import { useRunsReadAfterReconnect } from "./useRunsReadAfterReconnect";

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
  agent: LibraryAgent;
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
    hasError?: boolean;
  }) => void;
};

export function useSidebarRunsList({
  agent,
  onSelectRun,
  onCountsChange,
}: Args) {
  const graphId = agent.graph_id;
  const libraryAgentID = agent.id;
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

  const schedulerQueryOptions = useSchedulerQueryOptions();
  const schedulesQuery = useGetV1ListExecutionSchedulesForAGraph(graphId, {
    query: {
      enabled: !!graphId,
      select: okData,
      ...schedulerQueryOptions,
    },
  });

  const presetsQuery = useAgentPresetsQuery(graphId || undefined);

  const triggerAgentsEnabled = useGetFlag(Flag.GENERIC_TRIGGER_AGENTS);
  const triggerAgentsQuery = useGetV2ListTriggerAgents(libraryAgentID, {
    query: {
      enabled: triggerAgentsEnabled && !!libraryAgentID,
      select: okData,
      retry: retryUnlessClientError,
    },
  });

  const runs = useMemo(
    () => (runsQuery.data ? unpaginate(runsQuery.data, "executions") : []),
    [runsQuery.data],
  );

  const schedules = schedulesQuery.data || [];
  const allPresets = useMemo(
    () => presetsQuery.presets ?? [],
    [presetsQuery.presets],
  );
  const triggers = useMemo(
    () => allPresets.filter(isWebhookPreset),
    [allPresets],
  );
  const templates = useMemo(
    () => allPresets.filter((preset) => !isWebhookPreset(preset)),
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
  // While the scheduler restarts its schedules are unknown, and the runs,
  // templates and triggers still list.
  const schedulesUnavailable = useSchedulerOutage(schedulesQuery);
  const loading =
    !runsQuery.isSuccess ||
    (!schedulesQuery.isSuccess && !schedulesUnavailable) ||
    !presetsQuery.presetsSettled ||
    (triggerAgentsEnabled && !triggerAgentsQuery.isSuccess);
  const error =
    (schedulesUnavailable ? null : schedulesQuery.error) ||
    runsQuery.error ||
    presetsQuery.error ||
    (triggerAgentsEnabled ? triggerAgentsQuery.error : null);

  function retryFailedQueries() {
    if (runsQuery.isError) void runsQuery.refetch();
    if (schedulesQuery.isError) void schedulesQuery.refetch();
    if (presetsQuery.isError) void presetsQuery.refetch();
    if (triggerAgentsEnabled && triggerAgentsQuery.isError) {
      void triggerAgentsQuery.refetch();
    }
  }

  function retrySchedules() {
    void schedulesQuery.refetch();
  }

  // The key includes the graph ID, so this refetches only this graph's runs.
  function refetchRuns() {
    queryClient.invalidateQueries({ queryKey: runsQuery.queryKey });
  }

  useRunsReadAfterReconnect({
    queryKey: runsQuery.queryKey,
    onTasks: tabValue === "runs",
  });

  // Update query cache when execution events arrive via websocket
  useExecutionEvents({
    graphId: graphId || undefined,
    enabled: !!graphId && tabValue === "runs",
    // Refetching the infinite query reloads every loaded page in turn, so a
    // burst of events is folded into one refetch now and one at the end.
    throttleMs: EXECUTION_REFETCH_THROTTLE_MS,
    onExecutionUpdate: refetchRuns,
  });

  // Notify parent about counts and loading state
  useEffect(() => {
    if (onCountsChange) {
      onCountsChange({
        runsCount,
        schedulesCount,
        templatesCount,
        triggersCount,
        loading,
        hasError: !!error,
      });
    }
  }, [
    onCountsChange,
    runsCount,
    schedulesCount,
    templatesCount,
    triggersCount,
    loading,
    error,
  ]);

  useEffect(() => {
    if (runs.length > 0 && tabValue === "runs" && !activeItem) {
      onSelectRun(runs[0].id, "runs");
    }
  }, [runs, activeItem, tabValue, onSelectRun]);

  // Nothing selected and schedules to show, either because there are no runs
  // or because the Scheduled tab is open: select the first schedule.
  useEffect(() => {
    if (
      !activeItem &&
      schedules.length > 0 &&
      (runs.length === 0 || tabValue === "scheduled")
    ) {
      onSelectRun(schedules[0].id, "scheduled");
    }
  }, [activeItem, runs.length, schedules, tabValue, onSelectRun]);

  useEffect(() => {
    if (templates.length > 0 && tabValue === "templates" && !activeItem) {
      onSelectRun(templates[0].id, "templates");
    }
  }, [templates, activeItem, tabValue, onSelectRun]);

  useEffect(() => {
    if (tabValue !== "triggers" || activeItem) return;
    if (triggers.length > 0) {
      onSelectRun(
        activeItemParamFor("webhook-trigger", triggers[0].id),
        "triggers",
      );
    } else if (triggerAgents.length > 0) {
      onSelectRun(
        activeItemParamFor("trigger-agent", triggerAgents[0].id),
        "triggers",
      );
    }
  }, [triggers, triggerAgents, activeItem, tabValue, onSelectRun]);

  return {
    runs,
    schedules,
    schedulesUnavailable,
    retrySchedules,
    retryingSchedules: schedulesQuery.isFetching,
    templates,
    triggers,
    triggerAgents,
    error,
    loading,
    runsQuery,
    retryFailedQueries,
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
