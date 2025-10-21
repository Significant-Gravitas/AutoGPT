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
import { useAgentStore, buildAgentInfoMap } from "./store";

type AgentInfoMap = Map<
  string,
  { name: string; description: string; library_agent_id?: string }
>;

export function useAgentActivityDropdown() {
  const [isOpen, setIsOpen] = useState(false);

  const [api] = useState(() => new BackendAPI());

  const [notifications, setNotifications] = useState<NotificationState>({
    activeExecutions: [],
    recentCompletions: [],
    recentFailures: [],
    totalCount: 0,
  });

  const [isConnected, setIsConnected] = useState(false);
  const [agentInfoMap, setAgentInfoMap] = useState<AgentInfoMap>(new Map());

  const { agents, loadFromCache, refreshAll } = useAgentStore();

  const {
    data: executions,
    isSuccess: executionsSuccess,
    error: executionsError,
  } = useGetV1ListAllExecutions({
    query: { select: (res) => (res.status === 200 ? res.data : null) },
  });

  // Load cached immediately on mount
  useEffect(() => {
    loadFromCache();
    const timer = setTimeout(() => {
      void refreshAll().then(() => {
        const latest = useAgentStore.getState().agents;
        if (latest && latest.length) setAgentInfoMap(buildAgentInfoMap(latest));
      });
    }, 5000);
    return () => clearTimeout(timer);
  }, [loadFromCache, refreshAll]);

  // Build map whenever agents in store change
  useEffect(() => {
    if (agents && agents.length) setAgentInfoMap(buildAgentInfoMap(agents));
  }, [agents]);

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
