import { formatRelativeDuration } from "@/app/(platform)/library/hooks/executionHelpers";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { formatDistanceStrict } from "date-fns";

export function isLiveRun(run: GraphExecution | undefined) {
  return (
    run?.status === AgentExecutionStatus.RUNNING ||
    run?.status === AgentExecutionStatus.QUEUED
  );
}

export function formatRunDuration(
  run: GraphExecution,
  nowMs: number,
): string | null {
  const isLive = isLiveRun(run);

  if (!isLive && run.stats?.duration != null) {
    return formatDistanceStrict(0, run.stats.duration * 1000);
  }

  if (!run.started_at) return null;
  const start = new Date(run.started_at).getTime();
  if (Number.isNaN(start)) return null;

  // The clock is floored to the second, so it can lag a run that has just started.
  if (isLive) return formatRelativeDuration(Math.max(0, nowMs - start));

  const end = run.ended_at ? new Date(run.ended_at).getTime() : Number.NaN;
  if (Number.isNaN(end) || end < start) return null;
  return formatDistanceStrict(0, end - start);
}
