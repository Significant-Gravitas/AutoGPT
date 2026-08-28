"use client";

import {
  getGetV1ListAllExecutionsQueryKey,
  getGetV1UserCostSummaryQueryKey,
  useGetV1ListAllExecutions,
  useGetV1UserCostSummary,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { UserExecutionCostSummary } from "@/app/api/__generated__/models/userExecutionCostSummary";
import { okData } from "@/app/api/helpers";
import { useExecutionEvents } from "@/hooks/useExecutionEvents";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo } from "react";
import type { FleetSummary } from "../types";
import {
  isActive,
  isAgentScheduled,
  isFailed,
  SEVENTY_TWO_HOURS_MS,
} from "./executionHelpers";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { getTenantEntityKey } from "@/services/org-team/identity";
import { useOrgTeamStore } from "@/services/org-team/store";

function isRecentFailure(
  status: string,
  endedAt?: string | Date | null,
): boolean {
  if (!isFailed(status)) return false;
  if (!endedAt) return false;
  const ts =
    endedAt instanceof Date ? endedAt.getTime() : new Date(endedAt).getTime();
  return Date.now() - ts < SEVENTY_TWO_HOURS_MS;
}

function isRecentCompletion(
  status: string,
  endedAt?: string | Date | null,
): boolean {
  if (status !== AgentExecutionStatus.COMPLETED) return false;
  if (!endedAt) return false;
  const ts =
    endedAt instanceof Date ? endedAt.getTime() : new Date(endedAt).getTime();
  return Date.now() - ts < SEVENTY_TWO_HOURS_MS;
}

export function useLibraryFleetSummary(
  agents: LibraryAgent[],
): FleetSummary | undefined {
  const queryClient = useQueryClient();
  const activeOrgID = useOrgTeamStore((s) => s.activeOrgID);
  const activeTeamID = useOrgTeamStore((s) => s.activeTeamID);
  const isTenantReady = useOrgTeamStore((s) => s.isLoaded);

  const { data: executions, isSuccess } = useGetV1ListAllExecutions({
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

  // Authoritative monthly total comes from the server-side aggregator so it
  // stays correct above the 250-row /executions cap.
  const { data: costSummary } = useGetV1UserCostSummary(undefined, {
    query: {
      enabled: isTenantReady,
      queryKey: getTeamScopedQueryKey(
        getGetV1UserCostSummaryQueryKey(),
        activeOrgID,
        activeTeamID,
      ),
      select: (res) => res.data as UserExecutionCostSummary,
      staleTime: 60_000,
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

  const handleExecutionUpdate = useCallback(() => {
    queryClient.invalidateQueries({
      queryKey: getGetV1ListAllExecutionsQueryKey(),
    });
  }, [queryClient]);

  useExecutionEvents({
    graphScopes: graphScopes.length > 0 ? graphScopes : undefined,
    enabled: graphScopes.length > 0,
    onExecutionUpdate: handleExecutionUpdate,
  });

  return useMemo(() => {
    if (!isSuccess || !executions) return undefined;

    const agentsWithActiveExecution = new Set<string>();
    const agentsWithRecentFailure = new Set<string>();
    const agentsWithRecentCompletion = new Set<string>();

    for (const exec of executions) {
      const executionKey = getTenantEntityKey(
        exec.graph_id,
        exec.organization_id,
        exec.team_id,
      );
      if (isActive(exec.status)) {
        agentsWithActiveExecution.add(executionKey);
      }
      if (isRecentFailure(exec.status, exec.ended_at)) {
        agentsWithRecentFailure.add(executionKey);
      }
      if (isRecentCompletion(exec.status, exec.ended_at)) {
        agentsWithRecentCompletion.add(executionKey);
      }
    }

    // Authoritative server total; renders $0.00 briefly during initial load,
    // then updates to the real value. A local fallback would mis-bucket
    // cross-month executions (server buckets by createdAt, the local list
    // exposes only started_at) and produce a misleading flash.
    const monthlySpend = costSummary?.total_cents ?? 0;

    const summary: FleetSummary = {
      running: 0,
      error: 0,
      completed: 0,
      listening: 0,
      scheduled: 0,
      idle: 0,
      monthlySpend,
    };

    for (const agent of agents) {
      const agentKey = getTenantEntityKey(
        agent.graph_id,
        agent.organization_id,
        agent.team_id,
      );
      if (agentsWithActiveExecution.has(agentKey)) {
        summary.running += 1;
      } else if (agentsWithRecentFailure.has(agentKey)) {
        summary.error += 1;
      } else if (agent.has_external_trigger) {
        summary.listening += 1;
      } else if (isAgentScheduled(agent)) {
        summary.scheduled += 1;
      } else {
        summary.idle += 1;
      }
      // Parallel counter: mutually exclusive with running/error (which match
      // the sitrep priority order used by the "Recently completed" tab list)
      // but orthogonal to listening/scheduled/idle.
      if (
        !agentsWithActiveExecution.has(agentKey) &&
        !agentsWithRecentFailure.has(agentKey) &&
        agentsWithRecentCompletion.has(agentKey)
      ) {
        summary.completed += 1;
      }
    }

    return summary;
  }, [agents, executions, isSuccess, costSummary]);
}
