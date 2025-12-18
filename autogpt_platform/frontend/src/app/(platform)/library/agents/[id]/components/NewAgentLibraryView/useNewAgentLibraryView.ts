import { useGetV2GetLibraryAgent } from "@/app/api/__generated__/endpoints/library/library";
import { useGetV2GetASpecificPreset } from "@/app/api/__generated__/endpoints/presets/presets";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { okData } from "@/app/api/helpers";
import { useParams } from "next/navigation";
import { parseAsString, useQueryStates } from "nuqs";
import { useCallback, useEffect, useMemo, useState } from "react";

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

  const {
    data: agent,
    isSuccess,
    error,
  } = useGetV2GetLibraryAgent(agentId, {
    query: {
      select: okData<LibraryAgent>,
    },
  });

  const [{ activeItem, activeTab: activeTabRaw }, setQueryStates] =
    useQueryStates({
      activeItem: parseAsString,
      activeTab: parseAsString,
    });

  const activeTab = useMemo(() => parseTab(activeTabRaw), [activeTabRaw]);

  const {
    data: _template,
    isSuccess: isTemplateLoaded,
    isLoading: isTemplateLoading,
    error: templateError,
  } = useGetV2GetASpecificPreset(activeItem ?? "", {
    query: {
      enabled: Boolean(activeTab === "templates" && activeItem),
      select: okData<LibraryAgentPreset>,
    },
  });
  const activeTemplate =
    isTemplateLoaded &&
    activeTab === "templates" &&
    _template?.id === activeItem
      ? _template
      : null;

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

  // Show sidebar layout while loading or when there are items
  const showSidebarLayout = sidebarLoading || hasAnyItems;

  useEffect(() => {
    if (agent) {
      document.title = `${agent.name} - Library - AutoGPT Platform`;
    }
  }, [agent]);

  useEffect(() => {
    if (
      activeTab === "triggers" &&
      sidebarCounts.triggersCount === 0 &&
      !sidebarLoading
    ) {
      setQueryStates({
        activeTab: "runs",
      });
    }
  }, [activeTab, sidebarCounts.triggersCount, sidebarLoading, setQueryStates]);

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

  function handleSetActiveTab(
    tab: "runs" | "scheduled" | "templates" | "triggers",
  ) {
    setQueryStates({
      activeTab: tab,
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

  return {
    agentId: id,
    agent,
    ready: isSuccess,
    activeTemplate,
    isTemplateLoading,
    error: error || templateError,
    hasAnyItems,
    showSidebarLayout,
    activeItem,
    sidebarLoading,
    activeTab,
    setActiveTab: handleSetActiveTab,
    handleClearSelectedRun,
    handleCountsChange,
    handleSelectRun,
    onRunInitiated,
    onTriggerSetup,
    onScheduleCreated,
  };
}
