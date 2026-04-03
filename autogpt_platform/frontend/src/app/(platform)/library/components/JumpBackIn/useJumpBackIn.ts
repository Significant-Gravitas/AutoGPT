"use client";

import { useGetV1ListAllExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { okData } from "@/app/api/helpers";
import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";
import { useMemo } from "react";

function isActive(status: AgentExecutionStatus) {
  return (
    status === AgentExecutionStatus.RUNNING ||
    status === AgentExecutionStatus.QUEUED ||
    status === AgentExecutionStatus.REVIEW
  );
}

function formatDuration(startedAt: Date | string | null | undefined): string {
  if (!startedAt) return "";

  const start = new Date(startedAt);
  if (isNaN(start.getTime())) return "";

  const ms = Date.now() - start.getTime();
  if (ms < 0) return "";

  const sec = Math.floor(ms / 1000);
  if (sec < 5) return "a few seconds";
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ${sec % 60}s`;
  const hr = Math.floor(min / 60);
  return `${hr}h ${min % 60}m`;
}

function getStatusLabel(status: AgentExecutionStatus) {
  if (status === AgentExecutionStatus.RUNNING) return "Running";
  if (status === AgentExecutionStatus.QUEUED) return "Queued";
  if (status === AgentExecutionStatus.REVIEW) return "Awaiting approval";
  return "";
}

export function useJumpBackIn() {
  const { data: executions, isLoading: executionsLoading } =
    useGetV1ListAllExecutions({
      query: { select: okData },
    });

  const { agentInfoMap, isRefreshing: agentsLoading } = useLibraryAgents();

  const activeExecution = useMemo(() => {
    if (!executions) return null;

    const active = executions
      .filter((e) => isActive(e.status))
      .sort((a, b) => {
        const aTime = a.started_at ? new Date(a.started_at).getTime() : 0;
        const bTime = b.started_at ? new Date(b.started_at).getTime() : 0;
        return bTime - aTime;
      });

    return active[0] ?? null;
  }, [executions]);

  const enriched = useMemo(() => {
    if (!activeExecution) return null;

    const info = agentInfoMap.get(activeExecution.graph_id);
    return {
      id: activeExecution.id,
      agentName: info?.name ?? "Unknown Agent",
      libraryAgentId: info?.library_agent_id,
      status: activeExecution.status,
      statusLabel: getStatusLabel(activeExecution.status),
      duration: formatDuration(activeExecution.started_at),
    };
  }, [activeExecution, agentInfoMap]);

  return {
    execution: enriched,
    isLoading: executionsLoading || agentsLoading,
  };
}
