import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";

interface RunWithStats {
  status: string;
  stats?: { node_error_count?: number } | null;
}

// A failed step doesn't fail its run (backend/executor/manager.py), so a run
// can report COMPLETED while one of its steps errored.
export function getFailedStepCount(run: RunWithStats): number {
  return run.stats?.node_error_count ?? 0;
}

export function isCompletedWithErrors(run: RunWithStats): boolean {
  return (
    run.status === AgentExecutionStatus.COMPLETED && getFailedStepCount(run) > 0
  );
}

export function describeFailedSteps(count: number): string {
  return count === 1 ? "1 step failed" : `${count} steps failed`;
}
