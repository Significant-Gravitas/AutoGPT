import { useGetV2GetLibraryAgent } from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { useParams } from "next/navigation";
import { parseAsString, useQueryStates } from "nuqs";
import { useCallback, useEffect, useMemo, useState } from "react";

function parseTab(value: string | null): "runs" | "scheduled" | "templates" {
  if (value === "runs" || value === "scheduled" || value === "templates") {
    return value;
  }
  return "runs";
}

export function useNewAgentLibraryView() {
  const { id } = useParams();
  const agentId = id as string;
  const {
    data: response,
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

  useEffect(() => {
    if (!activeTabRaw) {
      setQueryStates({
        activeTab: "runs",
      });
    }
  }, [activeTabRaw, setQueryStates]);

  const [sidebarCounts, setSidebarCounts] = useState({
    runsCount: 0,
    schedulesCount: 0,
  });

  const [sidebarLoading, setSidebarLoading] = useState(true);

  const hasAnyItems = useMemo(
    () =>
      (sidebarCounts.runsCount ?? 0) > 0 ||
      (sidebarCounts.schedulesCount ?? 0) > 0,
    [sidebarCounts],
  );

  // Show sidebar layout while loading or when there are items
  const showSidebarLayout = sidebarLoading || hasAnyItems;

  useEffect(() => {
    if (response) {
      document.title = `${response.name} - Library - AutoGPT Platform`;
    }
  }, [response]);

  function handleSelectRun(id: string, tab?: "runs" | "scheduled") {
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

  function handleSetActiveTab(tab: "runs" | "scheduled" | "templates") {
    setQueryStates({
      activeTab: tab,
    });
  }

  const handleCountsChange = useCallback(
    (counts: {
      runsCount: number;
      schedulesCount: number;
      loading?: boolean;
    }) => {
      setSidebarCounts({
        runsCount: counts.runsCount,
        schedulesCount: counts.schedulesCount,
      });
      if (counts.loading !== undefined) {
        setSidebarLoading(counts.loading);
      }
    },
    [],
  );

  return {
    agentId: id,
    ready: isSuccess,
    error,
    agent: response,
    hasAnyItems,
    showSidebarLayout,
    activeItem,
    sidebarLoading,
    activeTab,
    setActiveTab: handleSetActiveTab,
    handleClearSelectedRun,
    handleCountsChange,
    handleSelectRun,
  };
}
