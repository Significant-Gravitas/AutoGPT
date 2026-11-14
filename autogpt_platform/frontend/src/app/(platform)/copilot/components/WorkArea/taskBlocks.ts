import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskArtifact } from "@/app/api/__generated__/models/taskArtifact";
import type { TaskEvidence } from "@/app/api/__generated__/models/taskEvidence";

/** The file that holds the task's full result, if it left one. */
export function documentOf(artifacts: TaskArtifact[]) {
  return (
    artifacts
      .filter(
        (artifact) =>
          artifact.kind === "DOCUMENT" &&
          !artifact.is_missing &&
          !!artifact.workspace_file_id,
      )
      .at(-1) ?? null
  );
}

/** The file a piece of proof points at, when it can still be opened. */
export function sourceOf(item: TaskEvidence, artifacts: TaskArtifact[]) {
  const file = artifacts.find((artifact) => artifact.id === item.artifact_id);
  return file && !file.is_missing && file.workspace_file_id ? file : null;
}

/** A link's site, for a row's second line: "cursor.com". */
export function hostOf(url: string | null | undefined) {
  if (!url) return null;
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return null;
  }
}

/** A settled task no longer has a "doing now" line worth showing. */
export function isSettled(status: string) {
  return status === "DONE" || status === "FAILED" || status === "CANCELLED";
}

/** The chat a task is worked in, when it is not the one being looked at. */
export function workChatOf(task: Task, currentSessionID: string | null) {
  const chat = task.work_session_id;
  return chat && chat !== currentSessionID ? chat : null;
}
