import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { NodeResolutionData } from "@/app/(platform)/build/stores/types";
import { beautifyString } from "@/lib/utils";
import { RJSFSchema } from "@rjsf/utils";
import { CustomNodeData } from "./CustomNode";

/**
 * Resolves the display title for a node using a 3-tier fallback:
 *
 * 1. `customized_name` — the user's manual rename (highest priority)
 * 2. `agent_name` (+ version) from `hardcodedValues` — the selected agent's
 *    display name, persisted by blocks like AgentExecutorBlock
 * 3. `data.title` — the generic block name (e.g. "Agent Executor")
 *
 * `customized_name` is the user's explicit rename via double-click; it lives in
 * node metadata. `agent_name` is the programmatic name of the agent graph
 * selected in the block's input form; it lives in `hardcodedValues` alongside
 * `graph_version`. These are distinct sources of truth — customized_name always
 * wins because it reflects deliberate user intent.
 */
export function getNodeDisplayTitle(data: CustomNodeData): string {
  if (data.metadata?.customized_name) {
    return data.metadata.customized_name as string;
  }

  const agentName = data.hardcodedValues?.agent_name as string | undefined;
  const graphVersion = data.hardcodedValues?.graph_version as
    | number
    | undefined;
  if (agentName) {
    return graphVersion != null ? `${agentName} v${graphVersion}` : agentName;
  }

  return data.title;
}

/**
 * Returns the formatted display title for rendering.
 * Agent names and custom names are shown as-is; generic block names get
 * beautified and have the trailing " Block" suffix stripped.
 */
export function formatNodeDisplayTitle(data: CustomNodeData): string {
  const title = getNodeDisplayTitle(data);
  const isAgentOrCustom = !!(
    data.metadata?.customized_name || data.hardcodedValues?.agent_name
  );
  return isAgentOrCustom
    ? title
    : beautifyString(title)
        .replace(/ Block$/, "")
        .trim();
}

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
