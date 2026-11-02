import { useGetV1ListAllExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";

import { okData } from "@/app/api/helpers";
import { useExecutionEvents } from "@/hooks/useExecutionEvents";
import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  NotificationState,
  categorizeExecutions,
  enrichExecutionWithAgentInfo,
  handleExecutionUpdate,
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

  // Get all graph IDs from agentInfoMap
  const graphIds = useMemo(
    () => Array.from(agentInfoMap.keys()),
    [agentInfoMap],
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

  const processedExecutionsRef = useRef<typeof executions>(undefined);
  useEffect(() => {
    if (
      executions &&
      executionsSuccess &&
      processedExecutionsRef.current !== executions
    ) {
      setNotifications(categorizeExecutions(executions, agentInfoMap));
      processedExecutionsRef.current = executions;
    }
  }, [executions, executionsSuccess, agentInfoMap]);

  // Subscribe to execution events for all graphs
  useExecutionEvents({
    graphIds: graphIds.length > 0 ? graphIds : undefined,
    enabled: graphIds.length > 0,
    onExecutionUpdate: handleExecutionEvent,
  });

  return {
    ...notifications,
    activeExecutions: notifications.activeExecutions.map((execution) =>
      enrichExecutionWithAgentInfo(execution, agentInfoMap),
    ),
    recentCompletions: notifications.recentCompletions.map((execution) =>
      enrichExecutionWithAgentInfo(execution, agentInfoMap),
    ),
    recentFailures: notifications.recentFailures.map((execution) =>
      enrichExecutionWithAgentInfo(execution, agentInfoMap),
    ),
    libraryError: libraryError && !isFetchingLibrary,
    retryLibrary,
    isReady: executionsSuccess,
    error: executionsError,
    isOpen,
    setIsOpen,
  };
}
