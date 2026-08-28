import {
  getGetV1ListAllExecutionsQueryKey,
  useGetV1ListAllExecutions,
} from "@/app/api/__generated__/endpoints/graphs/graphs";

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
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";

export function useAgentActivityDropdown() {
  const [isOpen, setIsOpen] = useState(false);
  const activeOrgID = useOrgTeamStore((s) => s.activeOrgID);
  const activeTeamID = useOrgTeamStore((s) => s.activeTeamID);
  const isTenantReady = useOrgTeamStore((s) => s.isLoaded);
  const { agents, agentInfoMap } = useLibraryAgents();

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
      enabled: isTenantReady,
      queryKey: getTeamScopedQueryKey(
        getGetV1ListAllExecutionsQueryKey(),
        activeOrgID,
        activeTeamID,
      ),
      select: okData,
    },
    request: getTenantRequestInit(activeOrgID, activeTeamID, isTenantReady),
  });

  const graphScopes = useMemo(
    () =>
      agents.map((agent) => ({
        graphId: agent.graph_id,
        organizationId: agent.organization_id ?? null,
        teamId: agent.team_id ?? null,
      })),
    [agents],
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

  // Process initial execution state when data loads
  // Use a ref to track if we've already processed to avoid infinite loops
  const processedExecutionsRef = useRef<string | null>(null);
  useEffect(() => {
    const executionKey = executions
      ? `${executions.length}-${executionsSuccess}`
      : null;

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
  }, [executions, executionsSuccess, agentInfoMap]);

  // Subscribe to execution events for all graphs
  useExecutionEvents({
    graphScopes: graphScopes.length > 0 ? graphScopes : undefined,
    enabled: graphScopes.length > 0,
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
