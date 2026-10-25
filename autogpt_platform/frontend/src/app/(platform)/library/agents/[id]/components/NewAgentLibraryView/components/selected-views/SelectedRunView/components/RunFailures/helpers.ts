import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { GraphExecutionWithNodes } from "@/app/api/__generated__/models/graphExecutionWithNodes";
import { beautifyString } from "@/lib/utils";

export interface FailedStep {
  id: string;
  blockID: string;
  error: string;
}

// Every step that errors records its message as an "error" output. Steps
// swept to FAILED because the whole run ended have none, so they're skipped.
export function getFailedSteps(
  run: GraphExecution | GraphExecutionWithNodes,
): FailedStep[] {
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

export function getRunError(run: GraphExecution | GraphExecutionWithNodes) {
  if (run.status !== AgentExecutionStatus.FAILED) return null;
  return run.stats?.error || null;
}

export function formatStepName(blockName: string) {
  return beautifyString(blockName)
    .replace(/ Block$/, "")
    .trim();
}
