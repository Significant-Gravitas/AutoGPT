import { useGetV1ListAllExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";

import BackendAPI from "@/lib/autogpt-server-api/client";
import type { GraphExecution, GraphID } from "@/lib/autogpt-server-api/types";
import { useCallback, useEffect, useState } from "react";
import * as Sentry from "@sentry/nextjs";
import {
  NotificationState,
  categorizeExecutions,
  handleExecutionUpdate,
} from "./helpers";
import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";

export function useAgentActivityDropdown() {
  const [isOpen, setIsOpen] = useState(false);
  const [api] = useState(() => new BackendAPI());
  const { agentInfoMap } = useLibraryAgents();

  const [notifications, setNotifications] = useState<NotificationState>({
    activeExecutions: [],
    recentCompletions: [],
    recentFailures: [],
    totalCount: 0,
  });

  const [isConnected, setIsConnected] = useState(false);

  const {
    data: executions,
    isSuccess: executionsSuccess,
    error: executionsError,
  } = useGetV1ListAllExecutions({
    query: { select: (res) => (res.status === 200 ? res.data : null) },
  });

  // Handle real-time execution updates
  const handleExecutionEvent = useCallback(
    (execution: GraphExecution) => {
      setNotifications((currentState) =>
        handleExecutionUpdate(currentState, execution, agentInfoMap),
      );
    },
    [agentInfoMap],
  );

  // Process initial execution state when data loads
  useEffect(() => {
    if (executions && executionsSuccess && agentInfoMap.size > 0) {
      const notifications = categorizeExecutions(executions, agentInfoMap);
      setNotifications(notifications);
    }
  }, [executions, executionsSuccess, agentInfoMap]);

  // Initialize WebSocket connection for real-time updates
  useEffect(() => {
    if (!agentInfoMap.size) return;

    const connectHandler = api.onWebSocketConnect(() => {
      setIsConnected(true);
      agentInfoMap.forEach((_, graphId) => {
        api.subscribeToGraphExecutions(graphId as GraphID).catch((error) => {
          Sentry.captureException(error, {
            tags: {
              graphId,
            },
          });
        });
      });
    });

    const disconnectHandler = api.onWebSocketDisconnect(() => {
      setIsConnected(false);
    });

    const messageHandler = api.onWebSocketMessage(
      "graph_execution_event",
      handleExecutionEvent,
    );

    api.connectWebSocket();

    return () => {
      connectHandler();
      disconnectHandler();
      messageHandler();
      api.disconnectWebSocket();
    };
  }, [api, handleExecutionEvent, agentInfoMap]);

  return {
    ...notifications,
    isConnected,
    isReady: executionsSuccess,
    error: executionsError,
    isOpen,
    setIsOpen,
  };
}
