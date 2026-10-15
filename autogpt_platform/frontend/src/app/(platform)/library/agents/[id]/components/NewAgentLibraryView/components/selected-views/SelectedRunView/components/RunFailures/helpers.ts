import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GetV1GetExecutionDetails200 } from "@/app/api/__generated__/models/getV1GetExecutionDetails200";
import { beautifyString } from "@/lib/utils";

export interface FailedStep {
  id: string;
  blockID: string;
  error: string;
}

// Every step that errors records its message as an "error" output. Steps
// swept to FAILED because the whole run ended have none, so they're skipped.
export function getFailedSteps(run: GetV1GetExecutionDetails200): FailedStep[] {
  if (!("node_executions" in run)) return [];
  return run.node_executions.flatMap((nodeExecution) => {
    const error = nodeExecution.output_data.error?.[0];
    if (nodeExecution.status !== AgentExecutionStatus.FAILED || error == null) {
      return [];
    }
    return [
      {
        id: nodeExecution.node_exec_id,
        blockID: nodeExecution.block_id,
        error: typeof error === "string" ? error : JSON.stringify(error),
      },
    ];
  });
}

export function getRunError(run: GetV1GetExecutionDetails200) {
  if (run.status !== AgentExecutionStatus.FAILED) return null;
  return run.stats?.error || null;
}

export function formatStepName(blockName: string) {
  return beautifyString(blockName)
    .replace(/ Block$/, "")
    .trim();
}
