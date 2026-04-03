import type { UIMessage } from "ai";

export const ORIGINAL_TITLE = "AutoGPT";

/**
 * Build the document title showing how many sessions are ready.
 * Returns the base title when count is 0.
 */
export function formatNotificationTitle(count: number): string {
  return count > 0
    ? `(${count}) AutoPilot is ready - ${ORIGINAL_TITLE}`
    : ORIGINAL_TITLE;
}

/**
 * Safely parse a JSON string (from localStorage) into a `Set<string>` of
 * session IDs. Returns an empty set for `null`, malformed, or non-array values.
 */
export function parseSessionIDs(raw: string | null | undefined): Set<string> {
  if (!raw) return new Set();
  try {
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed)
      ? new Set<string>(parsed.filter((v) => typeof v === "string"))
      : new Set();
  } catch {
    return new Set();
  }
}

/**
 * Check whether a refetchSession result indicates the backend still has an
 * active SSE stream for this session.
 */
export function hasActiveBackendStream(result: { data?: unknown }): boolean {
  const d = result.data;
  return (
    d != null &&
    typeof d === "object" &&
    "status" in d &&
    d.status === 200 &&
    "data" in d &&
    d.data != null &&
    typeof d.data === "object" &&
    "active_stream" in d.data &&
    !!d.data.active_stream
  );
}

/** Mark any in-progress tool parts as completed/errored so spinners stop. */
export function resolveInProgressTools(
  messages: UIMessage[],
  outcome: "completed" | "cancelled",
): UIMessage[] {
  return messages.map((msg) => ({
    ...msg,
    parts: msg.parts.map((part) =>
      "state" in part &&
      (part.state === "input-streaming" || part.state === "input-available")
        ? outcome === "cancelled"
          ? { ...part, state: "output-error" as const, errorText: "Cancelled" }
          : { ...part, state: "output-available" as const, output: "" }
        : part,
    ),
  }));
}

/**
 * Deduplicate messages by ID and by consecutive content fingerprint.
 *
 * ID dedup catches exact duplicates within the same source.
 * Content dedup only compares each assistant message to its **immediate
 * predecessor** — this catches hydration/stream boundary duplicates (where
 * the same content appears under different IDs) without accidentally
 * removing legitimately repeated assistant responses that are far apart.
 */
export function deduplicateMessages(messages: UIMessage[]): UIMessage[] {
  const seenIds = new Set<string>();
  let lastAssistantFingerprint = "";

  return messages.filter((msg) => {
    if (seenIds.has(msg.id)) return false;
    seenIds.add(msg.id);

    if (msg.role === "assistant") {
      const fingerprint = msg.parts
        .map(
          (p) =>
            ("text" in p && p.text) ||
            ("toolCallId" in p && p.toolCallId) ||
            "",
        )
        .join("|");

      // Only dedup if this assistant message is identical to the previous one
      if (fingerprint && fingerprint === lastAssistantFingerprint) return false;
      if (fingerprint) lastAssistantFingerprint = fingerprint;
    } else {
      // Reset on non-assistant messages so that identical assistant responses
      // separated by a user message (e.g. "Done!" → user → "Done!") are kept.
      lastAssistantFingerprint = "";
    }

    return true;
  });
}
