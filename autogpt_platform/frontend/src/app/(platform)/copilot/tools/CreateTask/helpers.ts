import type { ToolUIPart } from "ai";

interface CreatedTask {
  id: string;
  title: string;
}

function parseObject(output: unknown): Record<string, unknown> | null {
  if (typeof output === "string") {
    try {
      return parseObject(JSON.parse(output) as unknown);
    } catch {
      return null;
    }
  }
  if (output && typeof output === "object" && !Array.isArray(output)) {
    return output as Record<string, unknown>;
  }
  return null;
}

export function getCreatedTask(output: unknown): CreatedTask | null {
  const obj = parseObject(output);
  if (!obj || obj.type !== "task_updated") return null;
  const task = parseObject(obj.task);
  if (!task || typeof task.id !== "string" || !task.id) return null;
  const title = typeof task.title === "string" ? task.title.trim() : "";
  return { id: task.id, title: title || "Untitled task" };
}

/** A created task leaves the chain for its own pill; a running or failed call
 *  stays a chain row, so its model-facing text folds under the disclosure. */
export function isCreatedTaskPart(part: { type: string }) {
  if (part.type !== "tool-create_task") return false;
  const tool = part as ToolUIPart;
  return (
    tool.state === "output-available" && getCreatedTask(tool.output) !== null
  );
}
