export interface TaskKickoff {
  taskId: string;
  taskKey: string;
  title: string;
}

const TASK_KICKOFF_KIND = "task_kickoff";

function asString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

/** The task a work chat's opening message starts (see the backend's
 *  `kickoff_metadata`), or null for any other message. */
export function getTaskKickoff(metadata: unknown): TaskKickoff | null {
  if (!metadata || typeof metadata !== "object") return null;
  const meta = metadata as Record<string, unknown>;
  if (meta.kind !== TASK_KICKOFF_KIND) return null;
  const taskId = asString(meta.task_id);
  const taskKey = asString(meta.task_key);
  if (!taskId || !taskKey) return null;
  return { taskId, taskKey, title: asString(meta.task_title) ?? taskKey };
}
