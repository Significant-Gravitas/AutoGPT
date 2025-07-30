import { useGetV1GetAllExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV2ListLibraryAgents } from "@/app/api/__generated__/endpoints/library/library";

import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import BackendAPI from "@/lib/autogpt-server-api/client";
import type { GraphExecution, GraphID } from "@/lib/autogpt-server-api/types";
import { useCallback, useEffect, useState } from "react";
import * as Sentry from "@sentry/nextjs";
import {
  NotificationState,
  categorizeExecutions,
  handleExecutionUpdate,
} from "./helpers";

type AgentInfoMap = Map<
  string,
  { name: string; description: string; library_agent_id?: string }
>;

export function useAgentActivityDropdown() {
  const [api] = useState(() => new BackendAPI());

  const [notifications, setNotifications] = useState<NotificationState>({
    activeExecutions: [],
    recentCompletions: [],
    recentFailures: [],
    totalCount: 0,
  });

  const [isConnected, setIsConnected] = useState(false);
  const [agentInfoMap, setAgentInfoMap] = useState<AgentInfoMap>(new Map());

  const {
    data: agents,
    isSuccess: agentsSuccess,
    error: agentsError,
  } = useGetV2ListLibraryAgents();

  const {
    data: executions,
    isSuccess: executionsSuccess,
    error: executionsError,
  } = useGetV1GetAllExecutions();

  // Create a map of library agents
  useEffect(() => {
    if (agents && agentsSuccess) {
      // SafeCast: library agents loaded
      const libraryAgents = agents.data as LibraryAgentResponse;

      if (!libraryAgents.agents || !libraryAgents.agents.length) return;

      const agentMap = new Map<
        string,
        { name: string; description: string; library_agent_id?: string }
      >();

      libraryAgents.agents.forEach((agent) => {
        if (agent.graph_id && agent.id) {
          agentMap.set(agent.graph_id, {
            name: agent.name || `Agent ${agent.graph_id.slice(0, 8)}`,
            description: agent.description || "",
            library_agent_id: agent.id,
          });
        }
      });

      setAgentInfoMap(agentMap);
    }
  }, [agents, agentsSuccess]);

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
      const notifications = categorizeExecutions(executions.data, agentInfoMap);
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
    isReady: executionsSuccess && agentsSuccess,
    error: executionsError || agentsError,
  };
}
