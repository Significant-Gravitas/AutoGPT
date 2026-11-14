import { asObject, str } from "../../components/ToolChain/resultHelpers";

export interface CreatedTask {
  id: string;
  title: string;
}

/** The task a `create_task` call made, read from its result. */
export function createdTaskOf(output: unknown): CreatedTask | null {
  const data = asObject(output);
  if (!data || data.type !== "task_created") return null;
  const task = asObject(data.task);
  const id = task && str(task, "id");
  if (!task || !id) return null;
  return { id, title: str(task, "title")?.trim() || "Untitled task" };
}
