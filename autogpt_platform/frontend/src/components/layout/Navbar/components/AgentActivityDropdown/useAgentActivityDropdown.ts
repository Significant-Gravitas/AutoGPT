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
} from "./helpers";

export function useAgentActivityDropdown() {
  const [isOpen, setIsOpen] = useState(false);
  const { agentInfoMap } = useLibraryAgents();

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
    dataUpdatedAt: executionsUpdatedAt,
    refetch: refetchExecutions,
  } = useGetV1ListAllExecutions({
    // In the new layout this list only mounts when the account menu opens;
    // the cached page-load copy would miss every run started since.
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
      setNotifications((currentState) =>
        handleExecutionUpdate(currentState, execution, agentInfoMap),
      );
    },
    [agentInfoMap],
  );

  // Rebuild from the execution list whenever it is (re)fetched.
  // Use a ref to track if we've already processed to avoid infinite loops
  const processedExecutionsRef = useRef<number | null>(null);
  useEffect(() => {
    const executionKey = executions ? executionsUpdatedAt : null;

    if (
      executions &&
      executionsSuccess &&
      agentInfoMap.size > 0 &&
      processedExecutionsRef.current !== executionKey
    ) {
      const notifications = categorizeExecutions(executions, agentInfoMap);
      setNotifications(notifications);
      processedExecutionsRef.current = executionKey;
    }
  }, [executions, executionsSuccess, executionsUpdatedAt, agentInfoMap]);

  // Subscribe to execution events for all graphs
  useExecutionEvents({
    graphIds: graphIds.length > 0 ? graphIds : undefined,
    enabled: graphIds.length > 0,
    onExecutionUpdate: handleExecutionEvent,
    onSubscribed: () => void refetchExecutions(),
  });

  return {
    ...notifications,
    isReady: executionsSuccess,
    error: executionsError,
    isOpen,
    setIsOpen,
  };
}
