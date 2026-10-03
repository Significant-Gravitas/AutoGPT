import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";

export const SEVENTY_TWO_HOURS_MS = 72 * 60 * 60 * 1000;

// Match AgentActivityDropdown/helpers.tsx getExecutionDuration: only the first
// few seconds stay vague; after that show second-level precision (#9690 Part 2).
const SHORT_DURATION_THRESHOLD_SECONDS = 5;

// Shared scheduled-predicate used by fleet summary, sitrep, status-map, and the
// list filter so the four call sites stay in lockstep. Only the current user's
// actual schedule (`is_scheduled`) counts — `recommended_schedule_cron` is a
// creator-defined suggestion shared by every user, not an active schedule.
export function isAgentScheduled(agent: { is_scheduled?: boolean }): boolean {
  return !!agent.is_scheduled;
}

export function isActive(status: string): boolean {
  return (
    status === AgentExecutionStatus.RUNNING ||
    status === AgentExecutionStatus.QUEUED ||
    status === AgentExecutionStatus.REVIEW
  );
}

export function isFailed(status: string): boolean {
  return (
    status === AgentExecutionStatus.FAILED ||
    status === AgentExecutionStatus.TERMINATED
  );
}

export function toEndTime(exec: GraphExecutionMeta): number {
  if (!exec.ended_at) return 0;
  return exec.ended_at instanceof Date
    ? exec.ended_at.getTime()
    : new Date(exec.ended_at).getTime();
}

export function endedAfter(exec: GraphExecutionMeta, cutoff: number): boolean {
  if (!exec.ended_at) return false;
  return toEndTime(exec) > cutoff;
}

export function runningMessage(
  status: string,
  startedAt?: string | Date | null,
  nowMs: number = Date.now(),
): string {
  if (status === AgentExecutionStatus.QUEUED) return "Queued for execution";
  if (status === AgentExecutionStatus.REVIEW) return "Awaiting review";
  if (!startedAt) return "Currently executing";
  const startedMs =
    startedAt instanceof Date
      ? startedAt.getTime()
      : new Date(startedAt).getTime();
  return `Running for ${formatRelativeDuration(nowMs - startedMs)}`;
}

export function formatRelativeDuration(ms: number): string {
  const seconds = Math.floor(Math.max(0, ms) / 1000);
  if (seconds < SHORT_DURATION_THRESHOLD_SECONDS) return "a few seconds";
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  const hours = Math.floor(minutes / 60);
  const remainingMin = minutes % 60;
  if (hours < 24)
    return remainingMin > 0 ? `${hours}h ${remainingMin}m` : `${hours}h`;
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
}
