import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { AgentExecutionWithInfo } from "../../helpers";

type Args = {
  activeExecutions: AgentExecutionWithInfo[];
  recentCompletions: AgentExecutionWithInfo[];
  recentFailures: AgentExecutionWithInfo[];
};

export function getSortedExecutions({
  activeExecutions,
  recentCompletions,
  recentFailures,
}: Args) {
  const allExecutions = [
    ...activeExecutions.map((e) => ({
      ...e,
      type: "running" as const,
    })),
    ...recentCompletions.map((e) => ({
      ...e,
      type: "completed" as const,
    })),
    ...recentFailures.map((e) => ({ ...e, type: "failed" as const })),
  ];

  return allExecutions.sort((a, b) => {
    // Priority order: RUNNING > QUEUED > everything else
    const aIsRunning = a.status === AgentExecutionStatus.RUNNING;
    const bIsRunning = b.status === AgentExecutionStatus.RUNNING;
    const aIsQueued = a.status === AgentExecutionStatus.QUEUED;
    const bIsQueued = b.status === AgentExecutionStatus.QUEUED;

    // RUNNING agents always at the very top
    if (aIsRunning && !bIsRunning) return -1;
    if (!aIsRunning && bIsRunning) return 1;

    // If both are running, sort by most recent start time
    if (aIsRunning && bIsRunning) {
      if (!a.started_at || !b.started_at) return 0;
      return (
        new Date(b.started_at).getTime() - new Date(a.started_at).getTime()
      );
    }

    // QUEUED agents come second
    if (aIsQueued && !bIsQueued) return -1;
    if (!aIsQueued && bIsQueued) return 1;

    // If both are queued, sort by most recent start time
    if (aIsQueued && bIsQueued) {
      if (!a.started_at || !b.started_at) return 0;
      return (
        new Date(b.started_at).getTime() - new Date(a.started_at).getTime()
      );
    }

    // For everything else (completed, failed, etc.), sort by most recent end time
    const aTime = a.ended_at;
    const bTime = b.ended_at;

    if (!aTime || !bTime) return 0;
    return new Date(bTime).getTime() - new Date(aTime).getTime();
  });
}
