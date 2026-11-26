import type { DetachPreview } from "../../lib/api/expertManage";

/** What firing pauses: the expert's schedules and triggers, by name. */
export function firePauseLine(preview: DetachPreview) {
  const names = [...preview.schedule_names, ...preview.trigger_names];
  if (names.length === 0) return "Nothing is scheduled, so nothing pauses.";
  const count = names.length === 1 ? "1 routine pauses" : `${names.length} routines pause`;
  return `${count}: ${names.join(", ")}.`;
}

/**
 * The server refuses a reset with 409 while the expert has a turn running,
 * and that one is worth waiting out. Anything else just failed.
 */
export function resetOutcome(error: unknown): "busy" | "failed" {
  const status =
    typeof error === "object" && error !== null && "status" in error
      ? error.status
      : null;
  return status === 409 ? "busy" : "failed";
}
