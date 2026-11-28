import type { DetachPreview } from "../../lib/api/expertManage";

/** What firing pauses: the expert's schedules and triggers, by name. */
export function firePauseLine(preview: DetachPreview) {
  const names = [...preview.schedule_names, ...preview.trigger_names];
  if (names.length === 0) return "Nothing is scheduled, so nothing pauses.";
  const count = names.length === 1 ? "1 routine pauses" : `${names.length} routines pause`;
  return `${count}: ${names.join(", ")}.`;
}

function statusOf(error: unknown) {
  return typeof error === "object" && error !== null && "status" in error
    ? error.status
    : null;
}

/** The server refuses a reset with 409 while the expert has a turn running. */
export function resetOutcome(error: unknown): "busy" | "failed" {
  return statusOf(error) === 409 ? "busy" : "failed";
}

export function isGoneError(error: unknown) {
  return statusOf(error) === 404;
}
