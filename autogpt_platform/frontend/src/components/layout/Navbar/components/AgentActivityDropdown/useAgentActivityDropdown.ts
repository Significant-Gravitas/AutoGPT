import { useGetV1ListAllExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";

import { okData } from "@/app/api/helpers";
import { useExecutionEvents } from "@/hooks/useExecutionEvents";
import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  NotificationState,
  categorizeExecutions,
  handleExecutionUpdate,
  relabelNotifications,
} from "./helpers";

export function useAgentActivityDropdown() {
  const [isOpen, setIsOpen] = useState(false);
  const {
    agentInfoMap,
    isError: libraryError,
    isFetching: isFetchingLibrary,
    retry: retryLibrary,
  } = useLibraryAgents();

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
  } = useGetV1ListAllExecutions({
    query: { select: okData },
  });

  const graphIds = useMemo(
    () =>
      Array.from(
        new Set([
          ...(executions?.map((execution) => execution.graph_id) ?? []),
          ...agentInfoMap.keys(),
        ]),
      ),
    [executions, agentInfoMap],
  );

  // Handle real-time execution updates
  const handleExecutionEvent = useCallback(
    (execution: GraphExecution) => {
      setNotifications((currentState) =>
        handleExecutionUpdate(currentState, execution, agentInfoMap),
      );
    },
    [agentInfoMap],
  );

  const processedExecutionCount = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (
      executions &&
      executionsSuccess &&
      processedExecutionCount.current !== executions.length
    ) {
      setNotifications(categorizeExecutions(executions, agentInfoMap));
      processedExecutionCount.current = executions.length;
    }
  }, [executions, executionsSuccess, agentInfoMap]);

  // Subscribe to execution events for all graphs
  useExecutionEvents({
    graphIds: graphIds.length > 0 ? graphIds : undefined,
    enabled: graphIds.length > 0,
    onExecutionUpdate: handleExecutionEvent,
  });

  return {
    ...relabelNotifications(notifications, agentInfoMap),
    libraryError: libraryError && !isFetchingLibrary,
    retryLibrary,
    isReady: executionsSuccess,
    error: executionsError,
    isOpen,
    setIsOpen,
  };
}
