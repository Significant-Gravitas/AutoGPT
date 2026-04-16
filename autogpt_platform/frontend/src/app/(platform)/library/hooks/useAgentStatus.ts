"use client";

import { useMemo } from "react";
import { useGetV1ListAllExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";
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
  isFailed,
  toEndTime,
  SEVENTY_TWO_HOURS_MS,
} from "./executionHelpers";

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
    } else if (agent.recommended_schedule_cron) {
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
  const { data: executions } = useGetV1ListAllExecutions({
    query: { select: okData },
  });

  return useMemo(() => {
    const map = new Map<string, AgentStatusInfo>();
    const execsByGraph = new Map<string, GraphExecutionMeta[]>();

    for (const exec of executions ?? []) {
      const list = execsByGraph.get(exec.graph_id);
      if (list) {
        list.push(exec);
      } else {
        execsByGraph.set(exec.graph_id, [exec]);
      }
    }

    for (const agent of agents) {
      const agentExecs = execsByGraph.get(agent.graph_id) ?? [];
      map.set(agent.graph_id, computeAgentStatus(agent, agentExecs));
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
  graphID: string,
): AgentStatusInfo {
  return statusMap.get(graphID) ?? DEFAULT_STATUS;
}

export function useFleetSummary(agents: LibraryAgent[]): FleetSummary {
  const { data: executions } = useGetV1ListAllExecutions({
    query: { select: okData },
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
        if (isActive(exec.status)) {
          activeGraphIds.add(exec.graph_id);
        }
        const endedTs = exec.ended_at
          ? new Date(
              exec.ended_at instanceof Date
                ? exec.ended_at.getTime()
                : exec.ended_at,
            ).getTime()
          : 0;
        if (isFailed(exec.status) && endedTs > cutoff) {
          errorGraphIds.add(exec.graph_id);
        }
        if (
          exec.status === AgentExecutionStatus.COMPLETED &&
          endedTs > cutoff
        ) {
          completedGraphIds.add(exec.graph_id);
        }
      }
    }

    for (const agent of agents) {
      if (activeGraphIds.has(agent.graph_id)) {
        counts.running += 1;
      } else if (errorGraphIds.has(agent.graph_id)) {
        counts.error += 1;
      } else if (agent.has_external_trigger) {
        counts.listening += 1;
      } else if (agent.recommended_schedule_cron) {
        counts.scheduled += 1;
      } else {
        counts.idle += 1;
      }
      if (completedGraphIds.has(agent.graph_id)) {
        counts.completed += 1;
      }
    }

    return counts;
  }, [agents, executions]);
}

export { deriveHealth };
