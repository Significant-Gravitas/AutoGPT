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
