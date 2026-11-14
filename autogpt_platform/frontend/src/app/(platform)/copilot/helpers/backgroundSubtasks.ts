export function backgroundRefreshInterval(
  session: { background_pending?: boolean | null } | null | undefined,
) {
  return session?.background_pending ? 3000 : false;
}

export function backgroundNotice(metadata: Record<string, unknown> | null) {
  if (metadata?.background_task_stage !== "followup") return null;
  return typeof metadata.display_content === "string" &&
    metadata.display_content
    ? metadata.display_content
    : "Otto is following up.";
}

/** The part a finished task's handover is drawn from: the expert's own
 *  result and document, in place of the plain "following up" line. */
export const TASK_HANDOFF_PART_TYPE = "data-task-handoff";

/** The task whose result this follow-up hands over, when it left one. */
export function taskHandoffOf(metadata: Record<string, unknown> | null) {
  if (metadata?.background_task_stage !== "followup") return null;
  const taskID = metadata.task_handoff;
  return typeof taskID === "string" && taskID ? taskID : null;
}

/** The task id a handoff part carries. */
export function handoffTaskID(data: unknown) {
  if (!data || typeof data !== "object") return null;
  const taskID = (data as { taskID?: unknown }).taskID;
  return typeof taskID === "string" && taskID ? taskID : null;
}
