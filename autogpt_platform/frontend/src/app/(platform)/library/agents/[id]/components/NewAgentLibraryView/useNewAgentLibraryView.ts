import { useGetV2GetLibraryAgent } from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { useParams } from "next/navigation";
import { parseAsString, useQueryState } from "nuqs";
import { useCallback, useMemo, useState } from "react";

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

  const [runParam, setRunParam] = useQueryState("executionId", parseAsString);
  const selectedRun = runParam ?? undefined;

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

  function handleSelectRun(id: string) {
    setRunParam(id, { shallow: true });
  }

  function handleClearSelectedRun() {
    setRunParam(null, { shallow: true });
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
    selectedRun,
    sidebarLoading,
    handleClearSelectedRun,
    handleCountsChange,
    handleSelectRun,
  };
}
