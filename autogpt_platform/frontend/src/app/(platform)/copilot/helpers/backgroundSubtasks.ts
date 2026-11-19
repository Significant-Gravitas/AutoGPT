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

/** A run's result as it was when it was handed over: a repeating or
 *  watching task's record only holds its latest run, and clears it while
 *  the next one works, so each handover keeps its own. */
export interface HandoffSnapshot {
  result: string;
  documentTitle: string | null;
  documentFileID: string | null;
}

function snapshotOf(value: unknown): HandoffSnapshot | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const text = (key: string) => {
    const found = raw[key];
    return typeof found === "string" && found ? found : null;
  };
  const result = text("result");
  if (!result) return null;
  return {
    result,
    documentTitle: text("documentTitle") ?? text("document_title"),
    documentFileID: text("documentFileID") ?? text("document_file_id"),
  };
}

/** The result the follow-up saved with its handover, when it saved one. */
export function taskHandoffSnapshotOf(
  metadata: Record<string, unknown> | null,
) {
  return snapshotOf(metadata?.task_handoff_result);
}

/** The saved result a handoff part carries, when it carries one. */
export function handoffSnapshot(data: unknown) {
  if (!data || typeof data !== "object") return null;
  return snapshotOf((data as { snapshot?: unknown }).snapshot);
}
