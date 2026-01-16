import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { NodeResolutionData } from "@/app/(platform)/build/stores/nodeStore";
import { RJSFSchema } from "@rjsf/utils";

export const nodeStyleBasedOnStatus: Record<AgentExecutionStatus, string> = {
  INCOMPLETE: "ring-slate-300 bg-slate-300",
  QUEUED: " ring-blue-300 bg-blue-300",
  RUNNING: "ring-amber-300 bg-amber-300",
  REVIEW: "ring-yellow-300 bg-yellow-300",
  COMPLETED: "ring-green-300 bg-green-300",
  TERMINATED: "ring-orange-300 bg-orange-300 ",
  FAILED: "ring-red-300 bg-red-300",
};

/**
 * Merges schemas during resolution mode to include removed inputs/outputs
 * that still have connections, so users can see and delete them.
 */
export function mergeSchemaForResolution(
  currentSchema: Record<string, unknown>,
  newSchema: Record<string, unknown>,
  resolutionData: NodeResolutionData,
  type: "input" | "output",
): Record<string, unknown> {
  const newProps = (newSchema.properties as RJSFSchema) || {};
  const currentProps = (currentSchema.properties as RJSFSchema) || {};
  const mergedProps = { ...newProps };
  const incomp = resolutionData.incompatibilities;

  if (type === "input") {
    // Add back missing inputs that have connections
    incomp.missingInputs.forEach((inputName: string) => {
      if (currentProps[inputName]) {
        mergedProps[inputName] = currentProps[inputName];
      }
    });
    // Add back inputs with type mismatches (keep old type so connection works visually)
    incomp.inputTypeMismatches.forEach(
      (mismatch: { name: string; oldType: string; newType: string }) => {
        if (currentProps[mismatch.name]) {
          mergedProps[mismatch.name] = currentProps[mismatch.name];
        }
      },
    );
  } else {
    // Add back missing outputs that have connections
    incomp.missingOutputs.forEach((outputName: string) => {
      if (currentProps[outputName]) {
        mergedProps[outputName] = currentProps[outputName];
      }
    });
  }

  return {
    ...newSchema,
    properties: mergedProps,
  };
}
