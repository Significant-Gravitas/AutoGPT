import {
  getGetV1ListAllExecutionsQueryKey,
  useGetV1ListAllExecutions,
} from "@/app/api/__generated__/endpoints/graphs/graphs";

import { okData } from "@/app/api/helpers";
import {
  refetchAfterReconnect,
  useExecutionEvents,
} from "@/hooks/useExecutionEvents";
import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  NotificationState,
  categorizeExecutions,
  handleExecutionUpdate,
  keepActiveRunsTheListOmits,
  relabelNotifications,
} from "./helpers";
import { useLiveUpdateLog } from "./useLiveUpdateLog";

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
    query: { select: okData, refetchOnMount: "always" },
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

  // Rebuild from each fetched list (keyed on dataUpdatedAt), then replay the
  // live updates that arrived after that list's fetch started.
  const processedListRef = useRef<number | null>(null);
  useEffect(() => {
    if (
      !executions ||
      !executionsSuccess ||
      agentInfoMap.size === 0 ||
      processedListRef.current === dataUpdatedAt
    )
      return;
    const fromList = categorizeExecutions(executions, agentInfoMap);
    const newer = liveUpdates.updatesNewerThan(dataUpdatedAt);
    setNotifications((shown) =>
      newer.reduce(
        (state, execution) =>
          handleExecutionUpdate(state, execution, agentInfoMap),
        keepActiveRunsTheListOmits(shown, fromList, executions),
      ),
    );
    processedListRef.current = dataUpdatedAt;
  }, [executions, executionsSuccess, dataUpdatedAt, agentInfoMap, liveUpdates]);

  // Subscribe to execution events for all graphs
  useExecutionEvents({
    graphIds: graphIds.length > 0 ? graphIds : undefined,
    enabled: graphIds.length > 0,
    onExecutionUpdate: handleExecutionEvent,
    onReconnect: () =>
      refetchAfterReconnect(queryClient, getGetV1ListAllExecutionsQueryKey()),
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
