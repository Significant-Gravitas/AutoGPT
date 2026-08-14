import { isOutputType, type OutputType } from "../WorkOutputSheet/helpers";

export interface WorkRunMetadata {
  executionId: string;
  graphId: string;
  libraryAgentId: string | null;
  graphName: string;
  status: string;
  outputType: OutputType;
  outputKey: string | null;
}

export function isFailedRunStatus(status: string): boolean {
  return status.toUpperCase().includes("FAILED");
}

const RUN_METADATA_KIND = "expert_run";

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

/**
 * Read the structured run payload the backend rides on an assistant message's
 * ``metadata`` bag. Returns null for legacy posts (no metadata) so they keep
 * rendering as plain markdown.
 */
export function getWorkRunMetadata(value: unknown): WorkRunMetadata | null {
  if (!value || typeof value !== "object") return null;
  const meta = value as Record<string, unknown>;
  if (meta.kind !== RUN_METADATA_KIND) return null;

  const executionId = asString(meta.execution_id);
  const graphId = asString(meta.graph_id);
  if (!executionId || !graphId) return null;

  const outputType = isOutputType(meta.output_type)
    ? meta.output_type
    : "unknown";

  return {
    executionId,
    graphId,
    libraryAgentId: asString(meta.library_agent_id),
    graphName: asString(meta.graph_name) ?? "Workflow run",
    status: asString(meta.status) ?? "completed",
    outputType,
    outputKey: asString(meta.output_key),
  };
}

export function toPreview(text: string): string {
  return text
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/^>\s?/gm, "")
    .replace(/\*\*/g, "")
    .replace(/\s+/g, " ")
    .trim();
}
