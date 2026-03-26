import { useGetV1ListAllExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { useExecutionEvents } from "@/hooks/useExecutionEvents";
import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { GraphExecution, GraphID } from "@/lib/autogpt-server-api/types";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  NotificationState,
  categorizeExecutions,
  handleExecutionUpdate,
  isActiveExecution,
} from "./helpers";

type AgentInfo = {
  name: string;
  description: string;
  library_agent_id?: string;
};

const SEVENTY_TWO_HOURS_IN_MS = 72 * 60 * 60 * 1000;
const MAX_AGENT_INFO_LOOKUPS = 25;

function toAgentInfo(agent: LibraryAgent): AgentInfo {
  return {
    name:
      agent.name ||
      (agent.graph_id ? `Agent ${agent.graph_id.slice(0, 8)}` : "Agent"),
    description: agent.description || "",
    library_agent_id: agent.id,
  };
}

export function useAgentActivityDropdown() {
  const api = useBackendAPI();
  const [isOpen, setIsOpen] = useState(false);
  const { agentInfoMap } = useLibraryAgents();
  const [resolvedAgentInfoMap, setResolvedAgentInfoMap] = useState<
    Map<string, AgentInfo>
  >(new Map());

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
    query: {
      select: okData,
      refetchInterval: 5000,
      refetchIntervalInBackground: false,
    },
  });

  const combinedAgentInfoMap = useMemo(() => {
    if (resolvedAgentInfoMap.size === 0) {
      return agentInfoMap;
    }

    const merged = new Map<string, AgentInfo>();
    resolvedAgentInfoMap.forEach((value, key) => {
      merged.set(key, value);
    });
    agentInfoMap.forEach((value, key) => {
      merged.set(key, value);
    });

    return merged;
  }, [agentInfoMap, resolvedAgentInfoMap]);

  const missingGraphIds = useMemo(() => {
    if (!executions) {
      return [];
    }

    const cutoffTime = Date.now() - SEVENTY_TWO_HOURS_IN_MS;
    const ids = new Set<string>();

    for (const execution of executions) {
      const endedAt = execution.ended_at
        ? new Date(execution.ended_at).getTime()
        : null;
      const isRelevant =
        isActiveExecution(execution) ||
        (endedAt !== null && Number.isFinite(endedAt) && endedAt > cutoffTime);

      if (isRelevant && !combinedAgentInfoMap.has(execution.graph_id)) {
        ids.add(execution.graph_id);
      }
    }

    return Array.from(ids).slice(0, MAX_AGENT_INFO_LOOKUPS);
  }, [combinedAgentInfoMap, executions]);

  const graphIds = useMemo(
    () => Array.from(combinedAgentInfoMap.keys()),
    [combinedAgentInfoMap],
  );

  const handleExecutionEvent = useCallback(
    (execution: GraphExecution) => {
      setNotifications((currentState) =>
        handleExecutionUpdate(currentState, execution, combinedAgentInfoMap),
      );
    },
    [combinedAgentInfoMap],
  );

  useEffect(() => {
    if (!executions || !executionsSuccess) {
      return;
    }

    setNotifications(categorizeExecutions(executions, combinedAgentInfoMap));
  }, [combinedAgentInfoMap, executions, executionsSuccess]);

  useEffect(() => {
    if (missingGraphIds.length === 0) {
      return;
    }

    let isCancelled = false;

    async function resolveMissingAgents() {
      const results = await Promise.allSettled(
        missingGraphIds.map((graphId) =>
          api.getLibraryAgentByGraphID(graphId as GraphID),
        ),
      );

      if (isCancelled) {
        return;
      }

      setResolvedAgentInfoMap((currentMap) => {
        let didChange = false;
        const nextMap = new Map(currentMap);

        results.forEach((result, index) => {
          if (result.status !== "fulfilled" || !result.value.graph_id) {
            return;
          }

          const graphId = missingGraphIds[index];
          const nextInfo = toAgentInfo(result.value);
          const existingInfo = nextMap.get(graphId);

          if (
            existingInfo?.name === nextInfo.name &&
            existingInfo?.description === nextInfo.description &&
            existingInfo?.library_agent_id === nextInfo.library_agent_id
          ) {
            return;
          }

          nextMap.set(graphId, nextInfo);
          didChange = true;
        });

        return didChange ? nextMap : currentMap;
      });
    }

    resolveMissingAgents();

    return () => {
      isCancelled = true;
    };
  }, [api, missingGraphIds]);

  useExecutionEvents({
    graphIds: graphIds.length > 0 ? graphIds : undefined,
    enabled: graphIds.length > 0,
    onExecutionUpdate: handleExecutionEvent,
  });

  return {
    ...notifications,
    isReady: executionsSuccess,
    error: executionsError,
    isOpen,
    setIsOpen,
  };
}
