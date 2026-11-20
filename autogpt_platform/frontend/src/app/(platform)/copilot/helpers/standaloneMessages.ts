const STANDALONE_METADATA_KINDS = new Set([
  "expert_run",
  "scheduled_chat_notice",
]);

export function getStandaloneBubbleMetadata(
  metadata: unknown,
): Record<string, unknown> | null {
  if (!metadata || typeof metadata !== "object") return null;
  const kind = (metadata as Record<string, unknown>).kind;
  return typeof kind === "string" && STANDALONE_METADATA_KINDS.has(kind)
    ? (metadata as Record<string, unknown>)
    : null;
}
