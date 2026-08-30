export interface TaskCardMetadata {
  taskId: string;
  executionId: string;
  graphId: string;
  libraryAgentId: string | null;
  graphName: string;
  status: "DONE" | "FAILED";
}

const TASK_METADATA_KIND = "delegated_task";

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

/**
 * Read the task-outcome payload the backend rides on an assistant message's
 * ``metadata`` bag (see ``executor/task_outcomes.py``). Returns null for
 * anything else so other posts keep rendering as plain markdown.
 */
export function getTaskCardMetadata(value: unknown): TaskCardMetadata | null {
  if (!value || typeof value !== "object") return null;
  const meta = value as Record<string, unknown>;
  if (meta.kind !== TASK_METADATA_KIND) return null;

  const taskId = asString(meta.task_id);
  const executionId = asString(meta.execution_id);
  const graphId = asString(meta.graph_id);
  if (!taskId || !executionId || !graphId) return null;

  return {
    taskId,
    executionId,
    graphId,
    libraryAgentId: asString(meta.library_agent_id),
    graphName: asString(meta.graph_name) ?? "Agent task",
    // A card that can't read its own status must not claim success.
    status: meta.status === "DONE" ? "DONE" : "FAILED",
  };
}
