import {
  getGetV1ListAllExecutionsQueryKey,
  useGetV1ListAllExecutions,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";

import { okData } from "@/app/api/helpers";
import {
  refetchAfterReconnect,
  useExecutionEvents,
} from "@/hooks/useExecutionEvents";
import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createOmittedRunChecker } from "./checkOmittedRuns";
import {
  NotificationState,
  activeRunsTheListOmits,
  calculateTotalCount,
  categorizeExecutions,
  handleExecutionUpdate,
  keepActiveRunsTheListOmits,
  listHasEveryRun,
  relabelNotifications,
  removeExecutionFromAllCategories,
} from "./helpers";
import { listAllExecutionsQueryFn, useLiveUpdateLog } from "./useLiveUpdateLog";

export function useAgentActivityDropdown() {
  const [isOpen, setIsOpen] = useState(false);
  const { agentInfoMap } = useLibraryAgents();
  const queryClient = useQueryClient();
  const liveUpdates = useLiveUpdateLog();

  const [notifications, setNotifications] = useState<NotificationState>({
    activeExecutions: [],
    recentCompletions: [],
    recentFailures: [],
    totalCount: 0,
  });

  const {
    data: executions,
    isSuccess: executionsSuccess,
    error: executionsError,
    dataUpdatedAt,
  } = useGetV1ListAllExecutions({
    // The list mounts with the account menu, and a copy cached before the
    // menu opened can miss every run since.
    query: {
      select: okData,
      refetchOnMount: "always",
      queryFn: listAllExecutionsQueryFn,
    },
  });

  // Get all graph IDs from agentInfoMap
  const graphIds = useMemo(
    () => Array.from(agentInfoMap.keys()),
    [agentInfoMap],
  );

  // Handle real-time execution updates
  const handleExecutionEvent = useCallback(
    (execution: GraphExecution) => {
      liveUpdates.recordUpdate(execution);
      setNotifications((currentState) =>
        handleExecutionUpdate(currentState, execution, agentInfoMap),
      );
    },
    [agentInfoMap, liveUpdates],
  );

  // Rebuild from each fetched list (keyed on dataUpdatedAt), then replay what
  // came after that list's fetch started.
  const processedListRef = useRef<number | null>(null);
  const [omittedRuns] = useState(createOmittedRunChecker);
  // Only a reconnect loses live updates, so only the list read after one
  // reads left-out runs by ID.
  const listAfterReconnectRef = useRef(false);
  useEffect(() => {
    if (
      !executions ||
      !executionsSuccess ||
      agentInfoMap.size === 0 ||
      processedListRef.current === dataUpdatedAt
    )
      return;
    const fromList = categorizeExecutions(executions, agentInfoMap);
    const newer = liveUpdates.updatesNewerThan(
      dataUpdatedAt,
      executions,
      listHasEveryRun(executions),
    );
    // The runs to read come from the state this rebuild produces, not the
    // rendered one: a live update between the two could make the check read
    // a run that just ended, or skip one. The updater form keeps the
    // setState out of the effect body (see react-hooks/set-state-in-effect);
    // it returns the same state the check reads below.
    const rebuilt = newer.reduce(
      (state, execution) =>
        handleExecutionUpdate(state, execution, agentInfoMap),
      keepActiveRunsTheListOmits(notifications, fromList, executions),
    );
    setNotifications(() => rebuilt);
    processedListRef.current = dataUpdatedAt;
    const leftOut = activeRunsTheListOmits(rebuilt, executions);
    const readArgs = {
      log: liveUpdates,
      onRead: (run: GraphExecutionMeta) =>
        setNotifications((state) =>
          handleExecutionUpdate(state, run, agentInfoMap),
        ),
      onGone: (runID: string) =>
        setNotifications((state) =>
          calculateTotalCount(removeExecutionFromAllCategories(state, runID)),
        ),
    };
    if (listAfterReconnectRef.current) {
      listAfterReconnectRef.current = false;
      void omittedRuns.check({ runs: leftOut, ...readArgs });
      return;
    }
    // Later list refreshes carry the runs the reconnect check never got to
    // (past the first ten) and retry reads that failed, for those runs only.
    void omittedRuns.followUp({ runs: leftOut, ...readArgs });
  }, [
    executions,
    executionsSuccess,
    dataUpdatedAt,
    agentInfoMap,
    liveUpdates,
    omittedRuns,
  ]);

  // Subscribe to execution events for all graphs
  useExecutionEvents({
    graphIds: graphIds.length > 0 ? graphIds : undefined,
    enabled: graphIds.length > 0,
    onExecutionUpdate: handleExecutionEvent,
    onReconnect: () => {
      listAfterReconnectRef.current = true;
      refetchAfterReconnect(queryClient, getGetV1ListAllExecutionsQueryKey());
    },
  });

  return {
    // Named at read time: the library loads page by page, and runs of
    // workflows on later pages get their names once those pages land.
    ...relabelNotifications(notifications, agentInfoMap),
    isReady: executionsSuccess,
    error: executionsError,
    isOpen,
    setIsOpen,
  };
}
