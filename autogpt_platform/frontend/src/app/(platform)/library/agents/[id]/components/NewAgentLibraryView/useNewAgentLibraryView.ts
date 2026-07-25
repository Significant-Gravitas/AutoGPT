import {
  useGetV2GetLibraryAgent,
  useGetV2ListTriggerAgents,
} from "@/app/api/__generated__/endpoints/library/library";
import { useGetV2GetASpecificPreset } from "@/app/api/__generated__/endpoints/presets/presets";
import { useGetV1ListExecutionSchedulesForAGraph } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { okData } from "@/app/api/helpers";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { useParams } from "next/navigation";
import { parseAsString, useQueryStates } from "nuqs";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  deriveSelectedTriggerKind,
  parseActiveItemParam,
  retryUnlessClientError,
  SelectedTriggerKind,
} from "./helpers";
import { useAgentPresetsQuery } from "./hooks/useAgentPresetsQuery";

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

export function useNewAgentLibraryView() {
  const { id } = useParams();
  const agentId = id as string;

  // TODO(#12740 / autogpt-pr-reviewer): when agent.is_hidden is true,
  // surface a banner that this is a trigger agent and link to its parent.
  // Needs a back-derivation path (parent has no FK to its triggers) —
  // either a new endpoint or scanning AgentExecutorBlock constantInput
  // across the user's library.
  const {
    data: agent,
    isSuccess,
    error,
  } = useGetV2GetLibraryAgent(agentId, { query: { select: okData } });

  const { enabled: triggerAgentsEnabled, ready: triggerAgentsFlagReady } =
    useFlagStatus(Flag.GENERIC_TRIGGER_AGENTS);
  // This list is unpaginated on the backend, so membership is authoritative.
  const triggerAgentsQuery = useGetV2ListTriggerAgents(agentId, {
    query: {
      enabled: triggerAgentsEnabled && !!agentId,
      select: okData,
      retry: retryUnlessClientError,
    },
  });
  const triggerAgents = triggerAgentsQuery.data;

  const presetsQuery = useAgentPresetsQuery(agent?.graph_id);
  const presets = presetsQuery.data?.presets;
  const presetsComplete =
    !!presetsQuery.data &&
    presetsQuery.data.pagination.total_items <=
      presetsQuery.data.presets.length;

  const [{ activeItem, activeTab: activeTabRaw }, setQueryStates] =
    useQueryStates({
      activeItem: parseAsString,
      activeTab: parseAsString,
    });

  const activeTab = useMemo(() => parseTab(activeTabRaw), [activeTabRaw]);
  const { activeItemId, triggerKindHint } = parseActiveItemParam(activeItem);

  const onTemplatesTab = Boolean(activeTab === "templates" && activeItemId);
  const templateQuery = useGetV2GetASpecificPreset(activeItemId ?? "", {
    query: {
      enabled: onTemplatesTab,
      select: okData,
      retry: retryUnlessClientError,
    },
  });
  // This query shares its cache key with SelectedTriggerView's preset detail
  // fetch, so its state must stay scoped to the Templates tab — a preset 404
  // on the Triggers tab is handled inline there, not as a page-level error.
  const activeTemplate =
    onTemplatesTab && templateQuery.data?.id === activeItemId
      ? templateQuery.data
      : null;
  const isTemplateLoading = templateQuery.isLoading;
  const templateError = onTemplatesTab ? templateQuery.error : null;

  useEffect(() => {
    if (!activeTabRaw && !activeItem) {
      setQueryStates({
        activeTab: "runs",
      });
    }
  }, [activeTabRaw, activeItem, setQueryStates]);

  const [sidebarCounts, setSidebarCounts] = useState({
    runsCount: 0,
    schedulesCount: 0,
    templatesCount: 0,
    triggersCount: 0,
  });

  const [sidebarLoading, setSidebarLoading] = useState(true);

  const hasAnyItems = useMemo(
    () =>
      (sidebarCounts.runsCount ?? 0) > 0 ||
      (sidebarCounts.schedulesCount ?? 0) > 0 ||
      (sidebarCounts.templatesCount ?? 0) > 0 ||
      (sidebarCounts.triggersCount ?? 0) > 0,
    [sidebarCounts],
  );

  // Show sidebar layout while loading or when there are items or settings is selected
  const showSidebarLayout =
    sidebarLoading || hasAnyItems || activeItem === "settings";

  useEffect(() => {
    if (agent) {
      document.title = `${agent.name} - Library - AutoGPT Platform`;
    }
  }, [agent]);

  // Leave the Triggers tab when it becomes empty — but never while an item
  // is still selected: a stale selection must keep its detail pane (showing
  // the not-found state) instead of being re-routed to Runs as a bogus run ID.
  useEffect(() => {
    if (
      activeTab === "triggers" &&
      !activeItemId &&
      sidebarCounts.triggersCount === 0 &&
      !sidebarLoading
    ) {
      setQueryStates({
        activeTab: "runs",
      });
    }
  }, [
    activeTab,
    activeItemId,
    sidebarCounts.triggersCount,
    sidebarLoading,
    setQueryStates,
  ]);

  function handleSelectRun(
    id: string,
    tab?: "runs" | "scheduled" | "templates" | "triggers",
  ) {
    setQueryStates({
      activeItem: id,
      activeTab: tab ?? "runs",
    });
  }

  function handleClearSelectedRun() {
    setQueryStates({
      activeItem: null,
    });
  }

  const { data: schedules } = useGetV1ListExecutionSchedulesForAGraph(
    agent?.graph_id || "",
    {
      query: {
        enabled: !!agent?.graph_id,
        select: okData,
      },
    },
  );

  function handleScheduleDeleted(deletedScheduleId: string) {
    if (activeItem !== deletedScheduleId) {
      return;
    }

    if (!schedules) {
      handleClearSelectedRun();
      return;
    }

    const remainingSchedules = schedules.filter(
      (s) => s.id !== deletedScheduleId,
    );

    if (remainingSchedules.length > 0) {
      handleSelectRun(remainingSchedules[0].id, "scheduled");
    } else {
      handleClearSelectedRun();
    }
  }

  function handleSetActiveTab(
    tab: "runs" | "scheduled" | "templates" | "triggers",
  ) {
    setQueryStates({
      activeTab: tab,
    });
  }

  function handleSelectSettings() {
    setQueryStates({
      activeItem: "settings",
      activeTab: "runs", // Reset to runs tab when going to settings
    });
  }

  const handleCountsChange = useCallback(
    (counts: {
      runsCount: number;
      schedulesCount: number;
      templatesCount: number;
      triggersCount: number;
      loading?: boolean;
    }) => {
      setSidebarCounts({
        runsCount: counts.runsCount,
        schedulesCount: counts.schedulesCount,
        templatesCount: counts.templatesCount,
        triggersCount: counts.triggersCount,
      });
      if (counts.loading !== undefined) {
        setSidebarLoading(counts.loading);
      }
    },
    [],
  );

  function onItemCreated(
    createEvent:
      | { type: "runs"; item: GraphExecutionMeta }
      | { type: "triggers"; item: LibraryAgentPreset }
      | { type: "scheduled"; item: GraphExecutionJobInfo },
  ) {
    if (!hasAnyItems) {
      // Manually increment item count to flip hasAnyItems and showSidebarLayout
      const counts = {
        runsCount: createEvent.type === "runs" ? 1 : 0,
        triggersCount: createEvent.type === "triggers" ? 1 : 0,
        schedulesCount: createEvent.type === "scheduled" ? 1 : 0,
        templatesCount: 0,
      };
      handleCountsChange(counts);
    }
  }

  function onRunInitiated(newRun: GraphExecutionMeta) {
    if (!agent) return;
    onItemCreated({ item: newRun, type: "runs" });
  }

  function onTriggerSetup(newTrigger: LibraryAgentPreset) {
    if (!agent) return;
    onItemCreated({ item: newTrigger, type: "triggers" });
  }

  function onScheduleCreated(newSchedule: GraphExecutionJobInfo) {
    if (!agent) return;
    onItemCreated({ item: newSchedule, type: "scheduled" });
  }

  const triggerAgentsUnresolved =
    !triggerAgentsFlagReady ||
    (triggerAgentsEnabled && !triggerAgentsQuery.isSuccess);
  const selectedTriggerKind: SelectedTriggerKind | null =
    activeTab === "triggers"
      ? deriveSelectedTriggerKind({
          activeItemId,
          triggerKindHint,
          triggerAgents,
          presets,
          presetsComplete,
          listsResolved: presetsQuery.isSuccess && !triggerAgentsUnresolved,
          anyListFailed: presetsQuery.isError || triggerAgentsQuery.isError,
        })
      : null;
  function retryTriggerLists() {
    if (presetsQuery.isError) presetsQuery.refetch();
    if (triggerAgentsQuery.isError) triggerAgentsQuery.refetch();
  }

  return {
    agentId: id,
    agent,
    ready: isSuccess,
    activeTemplate,
    isTemplateLoading,
    error: error || templateError,
    hasAnyItems,
    showSidebarLayout,
    activeItemId,
    selectedTriggerKind,
    retryTriggerLists,
    sidebarLoading,
    activeTab,
    setActiveTab: handleSetActiveTab,
    handleClearSelectedRun,
    handleScheduleDeleted,
    handleCountsChange,
    handleSelectRun,
    handleSelectSettings,
    onRunInitiated,
    onTriggerSetup,
    onScheduleCreated,
  };
}
