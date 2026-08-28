"use client";

import { useMemo } from "react";
import {
  getGetV1ListAllExecutionsQueryKey,
  useGetV1ListAllExecutions,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import type {
  AgentStatus,
  AgentHealth,
  AgentStatusInfo,
  FleetSummary,
} from "../types";
import {
  isActive,
  isAgentScheduled,
  isFailed,
  toEndTime,
  SEVENTY_TWO_HOURS_MS,
} from "./executionHelpers";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { getTenantEntityKey } from "@/services/org-team/identity";
import { useOrgTeamStore } from "@/services/org-team/store";

function deriveHealth(
  status: AgentStatus,
  lastRunAt: string | null,
): AgentHealth {
  if (status === "error") return "attention";
  if (status === "idle" && lastRunAt) {
    const daysSince =
      (Date.now() - new Date(lastRunAt).getTime()) / (1000 * 60 * 60 * 24);
    if (daysSince > 14) return "stale";
  }
  return "good";
}

function computeAgentStatus(
  agent: LibraryAgent,
  agentExecutions: GraphExecutionMeta[],
): AgentStatusInfo {
  const activeExec = agentExecutions.find((e) => isActive(e.status));

  let status: AgentStatus;
  let lastError: string | null = null;
  let lastRunAt: string | null = null;
  const activeExecutionID = activeExec?.id ?? null;

  if (activeExec) {
    status = "running";
  } else {
    const cutoff = Date.now() - SEVENTY_TWO_HOURS_MS;
    const recentFailed = agentExecutions.find(
      (e) =>
        isFailed(e.status) &&
        e.ended_at &&
        new Date(
          e.ended_at instanceof Date ? e.ended_at.getTime() : e.ended_at,
        ).getTime() > cutoff,
    );

    if (recentFailed) {
      status = "error";
      lastError =
        (recentFailed.stats?.error as string) ??
        (recentFailed.stats?.activity_status as string) ??
        "Execution failed";
    } else if (agent.has_external_trigger) {
      status = "listening";
    } else if (isAgentScheduled(agent)) {
      status = "scheduled";
    } else {
      status = "idle";
    }
  }

  const completedExecs = agentExecutions.filter((e) => e.ended_at);
  if (completedExecs.length > 0) {
    const sorted = completedExecs.sort((a, b) => toEndTime(b) - toEndTime(a));
    const endedAt = sorted[0].ended_at;
    lastRunAt =
      endedAt instanceof Date ? endedAt.toISOString() : String(endedAt);
  }

  const totalRuns = agent.execution_count ?? agentExecutions.length;

  return {
    status,
    health: deriveHealth(status, lastRunAt),
    progress: null,
    totalRuns,
    lastRunAt,
    lastError,
    activeExecutionID,
    monthlySpend: 0,
    nextScheduledRun: null,
    triggerType: agent.has_external_trigger ? "webhook" : null,
  };
}

export function useAgentStatusMap(
  agents: LibraryAgent[],
): Map<string, AgentStatusInfo> {
  const activeOrgID = useOrgTeamStore((s) => s.activeOrgID);
  const activeTeamID = useOrgTeamStore((s) => s.activeTeamID);
  const isTenantReady = useOrgTeamStore((s) => s.isLoaded);
  const { data: executions } = useGetV1ListAllExecutions({
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

  return useMemo(() => {
    const map = new Map<string, AgentStatusInfo>();
    const execsByGraph = new Map<string, GraphExecutionMeta[]>();

    for (const exec of executions ?? []) {
      const executionKey = getTenantEntityKey(
        exec.graph_id,
        exec.organization_id,
        exec.team_id,
      );
      const list = execsByGraph.get(executionKey);
      if (list) {
        list.push(exec);
      } else {
        execsByGraph.set(executionKey, [exec]);
      }
    }

    for (const agent of agents) {
      const agentKey = getTenantEntityKey(
        agent.graph_id,
        agent.organization_id,
        agent.team_id,
      );
      const agentExecs = execsByGraph.get(agentKey) ?? [];
      map.set(agentKey, computeAgentStatus(agent, agentExecs));
    }

    return map;
  }, [agents, executions]);
}

const DEFAULT_STATUS: AgentStatusInfo = {
  status: "idle",
  health: "good",
  progress: null,
  totalRuns: 0,
  lastRunAt: null,
  lastError: null,
  activeExecutionID: null,
  monthlySpend: 0,
  nextScheduledRun: null,
  triggerType: null,
};

export function getAgentStatus(
  statusMap: Map<string, AgentStatusInfo>,
  agent: LibraryAgent,
): AgentStatusInfo {
  return (
    statusMap.get(
      getTenantEntityKey(agent.graph_id, agent.organization_id, agent.team_id),
    ) ?? DEFAULT_STATUS
  );
}

export function useFleetSummary(agents: LibraryAgent[]): FleetSummary {
  const activeOrgID = useOrgTeamStore((s) => s.activeOrgID);
  const activeTeamID = useOrgTeamStore((s) => s.activeTeamID);
  const isTenantReady = useOrgTeamStore((s) => s.isLoaded);
  const { data: executions } = useGetV1ListAllExecutions({
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

  return useMemo(() => {
    const counts: FleetSummary = {
      running: 0,
      error: 0,
      completed: 0,
      listening: 0,
      scheduled: 0,
      idle: 0,
      monthlySpend: 0,
    };

    const activeGraphIds = new Set<string>();
    const errorGraphIds = new Set<string>();
    const completedGraphIds = new Set<string>();

    if (executions) {
      const cutoff = Date.now() - SEVENTY_TWO_HOURS_MS;
      for (const exec of executions) {
        const executionKey = getTenantEntityKey(
          exec.graph_id,
          exec.organization_id,
          exec.team_id,
        );
        if (isActive(exec.status)) {
          activeGraphIds.add(executionKey);
        }
        const endedTs = exec.ended_at
          ? new Date(
              exec.ended_at instanceof Date
                ? exec.ended_at.getTime()
                : exec.ended_at,
            ).getTime()
          : 0;
        if (isFailed(exec.status) && endedTs > cutoff) {
          errorGraphIds.add(executionKey);
        }
        if (
          exec.status === AgentExecutionStatus.COMPLETED &&
          endedTs > cutoff
        ) {
          completedGraphIds.add(executionKey);
        }
      }
    }

    for (const agent of agents) {
      const agentKey = getTenantEntityKey(
        agent.graph_id,
        agent.organization_id,
        agent.team_id,
      );
      if (activeGraphIds.has(agentKey)) {
        counts.running += 1;
      } else if (errorGraphIds.has(agentKey)) {
        counts.error += 1;
      } else if (agent.has_external_trigger) {
        counts.listening += 1;
      } else if (isAgentScheduled(agent)) {
        counts.scheduled += 1;
      } else {
        counts.idle += 1;
      }
      if (completedGraphIds.has(agentKey)) {
        counts.completed += 1;
      }
    }

    return counts;
  }, [agents, executions]);
}

export { deriveHealth };
