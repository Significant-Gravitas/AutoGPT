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
  scheduledWithinMs?: number,
): SitrepItemData[] {
  const { data: executions } = useGetV1ListAllExecutions({
    query: { select: okData },
  });

  return useMemo(() => {
    if (agents.length === 0) return [];

    const graphIdToAgent = new Map(agents.map((a) => [a.graph_id, a]));
    const agentExecutions = groupByAgent(executions ?? [], graphIdToAgent);
    const items: SitrepItemData[] = [];
    const coveredAgentIds = new Set<string>();

    for (const [agent, execs] of agentExecutions) {
      const item = buildSitrepFromExecutions(agent, execs);
      if (item) {
        items.push(item);
        coveredAgentIds.add(agent.id);
      }
    }

    for (const agent of agents) {
      if (coveredAgentIds.has(agent.id)) continue;
      const configItem = buildSitrepFromConfig(agent, scheduledWithinMs);
      if (configItem) items.push(configItem);
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
  }, [agents, executions, maxItems, scheduledWithinMs]);
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

function buildSitrepFromConfig(
  agent: LibraryAgent,
  scheduledWithinMs?: number,
): SitrepItemData | null {
  if (agent.has_external_trigger) {
    return {
      id: `${agent.id}-listening`,
      agentID: agent.id,
      agentName: agent.name,
      priority: "listening",
      message: "Waiting for trigger event",
      status: "listening",
    };
  }

  if (agent.is_scheduled || agent.recommended_schedule_cron) {
    if (!isNextRunWithin(agent.next_scheduled_run, scheduledWithinMs)) {
      return null;
    }
    return {
      id: `${agent.id}-scheduled`,
      agentID: agent.id,
      agentName: agent.name,
      priority: "scheduled",
      message: formatNextRun(agent.next_scheduled_run),
      status: "scheduled",
    };
  }

  return null;
}

function isNextRunWithin(
  iso: string | undefined | null,
  windowMs: number | undefined,
): boolean {
  if (windowMs === undefined) return true;
  if (!iso) return false;
  const diff = new Date(iso).getTime() - Date.now();
  return diff <= windowMs;
}

function formatNextRun(iso: string | undefined | null): string {
  if (!iso) return "Has a scheduled run";
  const diff = new Date(iso).getTime() - Date.now();
  const minutes = Math.round(diff / 60_000);
  if (minutes <= 0) return "Scheduled to run soon";
  if (minutes < 60) return `Scheduled to run in ${minutes}m`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `Scheduled to run in ${hours}h`;
  const days = Math.round(hours / 24);
  return `Scheduled to run in ${days}d`;
}
