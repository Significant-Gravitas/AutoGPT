import { useGetV1GetAllExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV2GetMyAgents } from "@/app/api/__generated__/endpoints/store/store";
import { useGetV2ListLibraryAgents } from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import BackendAPI from "@/lib/autogpt-server-api/client";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { useCallback, useEffect, useState } from "react";
import {
  NotificationState,
  categorizeExecutions,
  createAgentInfoMap,
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

  // Get library agents using generated hook
  const {
    data: myAgentsResponse,
    isLoading: isAgentsLoading,
    error: agentsError,
  } = useGetV2GetMyAgents({
    query: {
      enabled: true,
    },
  });

  // Get library agents data to map graph_id to library_agent_id
  const {
    data: libraryAgentsResponse,
    isLoading: isLibraryAgentsLoading,
    error: libraryAgentsError,
  } = useGetV2ListLibraryAgents(
    {},
    {
      query: {
        enabled: true,
      },
    },
  );

  // Get all executions using generated hook
  const {
    data: executionsResponse,
    isLoading: isExecutionsLoading,
    error: executionsError,
  } = useGetV1GetAllExecutions({
    query: {
      enabled: true,
    },
  });

  // Update agent info map when both agent data sources change
  useEffect(() => {
    if (
      myAgentsResponse?.data?.agents &&
      libraryAgentsResponse?.data &&
      "agents" in libraryAgentsResponse.data
    ) {
      const agentMap = createAgentInfoMap(myAgentsResponse.data.agents);

      // Add library agent ID mapping
      libraryAgentsResponse.data.agents.forEach(
        (libraryAgent: LibraryAgent) => {
          const existingInfo = agentMap.get(libraryAgent.graph_id);
          if (existingInfo) {
            agentMap.set(libraryAgent.graph_id, {
              ...existingInfo,
              library_agent_id: libraryAgent.id,
            });
          }
        },
      );

      setAgentInfoMap(agentMap);
    }
  }, [myAgentsResponse, libraryAgentsResponse]);

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
    if (
      executionsResponse?.data &&
      !isExecutionsLoading &&
      agentInfoMap.size > 0
    ) {
      const newNotifications = categorizeExecutions(
        executionsResponse.data,
        agentInfoMap,
      );

      setNotifications(newNotifications);
    }
  }, [executionsResponse, isExecutionsLoading, agentInfoMap]);

  // Initialize WebSocket connection for real-time updates
  useEffect(() => {
    const connectHandler = api.onWebSocketConnect(() => {
      setIsConnected(true);

      // Subscribe to graph executions for all user agents
      if (myAgentsResponse?.data?.agents) {
        myAgentsResponse.data.agents.forEach((agent) => {
          api
            .subscribeToGraphExecutions(agent.agent_id as any)
            .catch((error) => {
              console.error(
                `[AgentNotifications] Failed to subscribe to graph ${agent.agent_id}:`,
                error,
              );
            });
        });
      }
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
  }, [api, handleExecutionEvent, myAgentsResponse]);

  return {
    ...notifications,
    isConnected,
    isLoading: isAgentsLoading || isExecutionsLoading || isLibraryAgentsLoading,
    error: agentsError || executionsError || libraryAgentsError,
  };
}
