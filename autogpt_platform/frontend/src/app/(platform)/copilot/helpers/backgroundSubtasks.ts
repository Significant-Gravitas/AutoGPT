// The API client revives Z-suffixed timestamps into Dates and leaves the
// backend's +00:00 form as text, so both shapes reach this.
export function backgroundRefreshInterval(
  session:
    | { background_pending_until?: Date | string | null }
    | null
    | undefined,
  now = Date.now(),
) {
  const until = session?.background_pending_until;
  const at =
    until instanceof Date
      ? until.getTime()
      : typeof until === "string"
        ? Date.parse(until)
        : NaN;
  return at > now ? 3000 : false;
}

export function backgroundNotice(metadata: Record<string, unknown> | null) {
  if (metadata?.background_task_stage !== "followup") return null;
  return typeof metadata.display_content === "string" &&
    metadata.display_content
    ? metadata.display_content
    : "Otto is following up.";
}
