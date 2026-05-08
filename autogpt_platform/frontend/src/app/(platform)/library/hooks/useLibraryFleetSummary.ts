"use client";

import {
  getGetV1ListAllExecutionsQueryKey,
  useGetV1ListAllExecutions,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { useExecutionEvents } from "@/hooks/useExecutionEvents";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo } from "react";
import type { FleetSummary } from "../types";
import { isActive, isFailed, SEVENTY_TWO_HOURS_MS } from "./executionHelpers";

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

function toTimestamp(value?: string | Date | null): number | null {
  if (!value) return null;
  const ts =
    value instanceof Date ? value.getTime() : new Date(value).getTime();
  return Number.isFinite(ts) ? ts : null;
}

function startOfCurrentMonth(now: Date = new Date()): number {
  return Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1);
}

function currentDayKey(now: Date = new Date()): number {
  return Math.floor(now.getTime() / 86_400_000);
}

export function useLibraryFleetSummary(
  agents: LibraryAgent[],
): FleetSummary | undefined {
  const queryClient = useQueryClient();

  const { data: executions, isSuccess } = useGetV1ListAllExecutions({
    query: { select: okData },
  });

  const graphIDs = useMemo(() => agents.map((a) => a.graph_id), [agents]);

  // Recompute when the UTC day rolls over so `monthStart` stays fresh on a
  // long-open tab — read on every render; cheap. React Query refetches keep
  // the memo re-evaluating in practice.
  const dayKey = currentDayKey();

  const handleExecutionUpdate = useCallback(() => {
    queryClient.invalidateQueries({
      queryKey: getGetV1ListAllExecutionsQueryKey(),
    });
  }, [queryClient]);

  useExecutionEvents({
    graphIds: graphIDs.length > 0 ? graphIDs : undefined,
    enabled: graphIDs.length > 0,
    onExecutionUpdate: handleExecutionUpdate,
  });

  return useMemo(() => {
    if (!isSuccess || !executions) return undefined;

    const agentsWithActiveExecution = new Set<string>();
    const agentsWithRecentFailure = new Set<string>();
    const agentsWithRecentCompletion = new Set<string>();
    const monthStart = startOfCurrentMonth();
    let monthlySpendCents = 0;

    for (const exec of executions) {
      if (isActive(exec.status)) {
        agentsWithActiveExecution.add(exec.graph_id);
      }
      if (isRecentFailure(exec.status, exec.ended_at)) {
        agentsWithRecentFailure.add(exec.graph_id);
      }
      if (isRecentCompletion(exec.status, exec.ended_at)) {
        agentsWithRecentCompletion.add(exec.graph_id);
      }

      const startedTs = toTimestamp(exec.started_at);
      if (startedTs !== null && startedTs >= monthStart) {
        const cost = exec.stats?.cost;
        if (typeof cost === "number" && Number.isFinite(cost)) {
          monthlySpendCents += cost;
        }
      }
    }

    const summary: FleetSummary = {
      running: 0,
      error: 0,
      completed: 0,
      listening: 0,
      scheduled: 0,
      idle: 0,
      monthlySpend: monthlySpendCents,
    };

    for (const agent of agents) {
      if (agentsWithActiveExecution.has(agent.graph_id)) {
        summary.running += 1;
      } else if (agentsWithRecentFailure.has(agent.graph_id)) {
        summary.error += 1;
      } else if (agent.has_external_trigger) {
        summary.listening += 1;
      } else if (agent.is_scheduled || agent.recommended_schedule_cron) {
        summary.scheduled += 1;
      } else {
        summary.idle += 1;
      }
      // Parallel counter: mutually exclusive with running/error (which match
      // the sitrep priority order used by the "Recently completed" tab list)
      // but orthogonal to listening/scheduled/idle.
      if (
        !agentsWithActiveExecution.has(agent.graph_id) &&
        !agentsWithRecentFailure.has(agent.graph_id) &&
        agentsWithRecentCompletion.has(agent.graph_id)
      ) {
        summary.completed += 1;
      }
    }

    return summary;
  }, [agents, executions, isSuccess, dayKey]);
}
