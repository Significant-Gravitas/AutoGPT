"use client";

import { useGetV1ListAllExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { useMemo } from "react";
import type { SitrepItemData, SitrepPriority } from "../../types";
import {
  isActive,
  isFailed,
  toEndTime,
  endedAfter,
  runningMessage,
  SEVENTY_TWO_HOURS_MS,
} from "../../hooks/executionHelpers";

export function useSitrepItems(
  agents: LibraryAgent[],
  maxItems: number,
): SitrepItemData[] {
  const { data: executions } = useGetV1ListAllExecutions({
    query: { select: okData },
  });

  return useMemo(() => {
    if (!executions || agents.length === 0) return [];

    const graphIdToAgent = new Map(agents.map((a) => [a.graph_id, a]));
    const agentExecutions = groupByAgent(executions, graphIdToAgent);
    const items: SitrepItemData[] = [];

    for (const [agent, execs] of agentExecutions) {
      const item = buildSitrepFromExecutions(agent, execs);
      if (item) items.push(item);
    }

    const order: Record<SitrepPriority, number> = {
      error: 0,
      running: 1,
      stale: 2,
      success: 3,
      listening: 4,
      scheduled: 5,
      idle: 6,
    };
    items.sort((a, b) => order[a.priority] - order[b.priority]);

    return items.slice(0, maxItems);
  }, [agents, executions, maxItems]);
}

function groupByAgent(
  executions: GraphExecutionMeta[],
  graphIdToAgent: Map<string, LibraryAgent>,
): Map<LibraryAgent, GraphExecutionMeta[]> {
  const map = new Map<LibraryAgent, GraphExecutionMeta[]>();

  for (const exec of executions) {
    const agent = graphIdToAgent.get(exec.graph_id);
    if (!agent) continue;
    const list = map.get(agent);
    if (list) {
      list.push(exec);
    } else {
      map.set(agent, [exec]);
    }
  }

  return map;
}

function buildSitrepFromExecutions(
  agent: LibraryAgent,
  executions: GraphExecutionMeta[],
): SitrepItemData | null {
  const active = executions.find((e) => isActive(e.status));
  if (active) {
    return {
      id: `${agent.id}-${active.id}`,
      agentID: agent.id,
      agentName: agent.name,
      executionID: active.id,
      priority: "running",
      message:
        active.stats?.activity_status ??
        runningMessage(active.status, active.started_at),
      status: "running",
    };
  }

  const cutoff = Date.now() - SEVENTY_TWO_HOURS_MS;
  const recent = executions
    .filter((e) => endedAfter(e, cutoff))
    .sort((a, b) => toEndTime(b) - toEndTime(a));

  const lastFailed = recent.find((e) => isFailed(e.status));
  if (lastFailed) {
    const errorMsg =
      lastFailed.stats?.error ??
      lastFailed.stats?.activity_status ??
      "Execution failed";
    return {
      id: `${agent.id}-${lastFailed.id}`,
      agentID: agent.id,
      agentName: agent.name,
      executionID: lastFailed.id,
      priority: "error",
      message: typeof errorMsg === "string" ? errorMsg : "Execution failed",
      status: "error",
    };
  }

  const lastCompleted = recent.find(
    (e) => e.status === AgentExecutionStatus.COMPLETED,
  );
  if (lastCompleted) {
    const summary =
      lastCompleted.stats?.activity_status ?? "Completed successfully";
    return {
      id: `${agent.id}-${lastCompleted.id}`,
      agentID: agent.id,
      agentName: agent.name,
      executionID: lastCompleted.id,
      priority: "success",
      message: typeof summary === "string" ? summary : "Completed successfully",
      status: "idle",
    };
  }

  return null;
}
