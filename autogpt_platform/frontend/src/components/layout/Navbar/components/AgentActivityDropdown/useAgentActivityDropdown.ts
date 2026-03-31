import { useGetV1ListAllExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { okData } from "@/app/api/helpers";
import { useExecutionEvents } from "@/hooks/useExecutionEvents";
import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { GraphExecution, GraphID } from "@/lib/autogpt-server-api/types";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
const MAX_LOOKUP_FAILURES = 3;

function toAgentInfo(agent: {
  name: string;
  graph_id: string;
  description: string;
  id: string;
}): AgentInfo {
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
  const failedLookups = useRef<Map<string, number>>(new Map());
  const prevMissingIdsKey = useRef<string>("");

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

      if (
        isRelevant &&
        !combinedAgentInfoMap.has(execution.graph_id) &&
        (failedLookups.current.get(execution.graph_id) ?? 0) <
          MAX_LOOKUP_FAILURES
      ) {
        ids.add(execution.graph_id);
      }
    }

    const candidate = Array.from(ids).slice(0, MAX_AGENT_INFO_LOOKUPS);
    const key = candidate.join(",");

    // Stabilize reference: only return a new array when the actual IDs change
    if (key === prevMissingIdsKey.current) {
      return candidate.length === 0 ? [] : candidate;
    }
    prevMissingIdsKey.current = key;
    return candidate;
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
          const graphId = missingGraphIds[index];

          if (result.status !== "fulfilled" || !result.value.graph_id) {
            // Track failed lookups to prevent infinite retries
            const count = failedLookups.current.get(graphId) ?? 0;
            failedLookups.current.set(graphId, count + 1);
            return;
          }

          // Clear failure count on success
          failedLookups.current.delete(graphId);

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
