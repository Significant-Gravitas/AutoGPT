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

  // Live: started_at → now. Finished without stats: started_at → ended_at.
  if (!run.started_at) return null;
  const start = new Date(run.started_at).getTime();
  if (Number.isNaN(start)) return null;
  const endedMs = run.ended_at ? new Date(run.ended_at).getTime() : null;
  const end = isLive ? nowMs : endedMs;
  if (end === null || Number.isNaN(end) || end < start) return null;
  return formatDistanceStrict(0, end - start);
}
