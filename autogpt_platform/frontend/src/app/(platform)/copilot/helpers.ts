import { getSystemHeaders } from "@/lib/impersonation";
import { getWebSocketToken } from "@/lib/supabase/actions";
import type { UIMessage } from "ai";

export const ORIGINAL_TITLE = "AutoGPT";

/**
 * Returns HTTP headers required for direct backend requests from copilot:
 * - Authorization Bearer token (JWT)
 * - X-Act-As-User-Id impersonation header (if an admin is impersonating a user)
 *
 * Use this for all direct-to-backend fetch/SSE calls so that admin user
 * impersonation works consistently across the entire copilot feature.
 */
export async function getCopilotAuthHeaders(): Promise<Record<string, string>> {
  const { token, error } = await getWebSocketToken();
  if (error || !token) {
    console.warn("[Copilot] Failed to get auth token:", error);
    throw new Error("Authentication failed — please sign in again.");
  }
  return {
    Authorization: `Bearer ${token}`,
    ...getSystemHeaders(),
  };
}

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
 * Extract the user-visible text from the arguments passed to `sendMessage`.
 * Handles both `sendMessage("hello")` and `sendMessage({ text: "hello" })`.
 */
export function extractSendMessageText(firstArg: unknown): string {
  if (firstArg && typeof firstArg === "object" && "text" in firstArg)
    return (firstArg as { text: string }).text;
  return String(firstArg ?? "");
}

interface SuppressDuplicateArgs {
  text: string;
  isReconnectScheduled: boolean;
  lastSubmittedText: string | null;
  messages: UIMessage[];
}

/**
 * Reason a sendMessage was suppressed, or ``null`` to pass through.
 *
 * - ``"reconnecting"``: the stream is reconnecting; the caller should
 *   notify the user (the UI may not yet reflect the disabled state).
 * - ``"duplicate"``: the same text was just submitted and echoed back
 *   by the session — safe to silently drop (user double-clicked).
 */
export type SuppressReason = "reconnecting" | "duplicate" | null;

/**
 * Determine whether a sendMessage call should be suppressed to prevent
 * duplicate POSTs during reconnect cycles or re-submits of the same text.
 *
 * Returns the reason so callers can surface user-visible feedback when
 * the suppression isn't just a silent duplicate.
 */
export function getSendSuppressionReason({
  text,
  isReconnectScheduled,
  lastSubmittedText,
  messages,
}: SuppressDuplicateArgs): SuppressReason {
  if (isReconnectScheduled) return "reconnecting";

  if (text && lastSubmittedText === text) {
    const lastUserMsg = messages.filter((m) => m.role === "user").pop();
    const lastUserText = lastUserMsg?.parts
      ?.map((p) => ("text" in p ? p.text : ""))
      .join("")
      .trim();
    if (lastUserText === text) return "duplicate";
  }

  return null;
}

/**
 * Backwards-compatible boolean wrapper for ``getSendSuppressionReason``.
 *
 * @deprecated Call ``getSendSuppressionReason`` directly so callers can
 * distinguish between reconnect and duplicate suppression.
 */
export function shouldSuppressDuplicateSend(
  args: SuppressDuplicateArgs,
): boolean {
  return getSendSuppressionReason(args) !== null;
}

/**
 * Deduplicate messages by ID and by content fingerprint.
 *
 * ID dedup catches exact duplicates within the same source.
 * Content dedup uses a composite key of `role + preceding-user-message-id +
 * content-fingerprint` to detect replayed messages that arrive with new
 * IDs after an SSE reconnection replays from the beginning of the Redis
 * stream. Scoping by user message ID (not text) preserves the second
 * assistant reply when the user asks the same question twice and gets the
 * same answer — two different user messages produce two different IDs even
 * when their text is identical.
 */
export function deduplicateMessages(messages: UIMessage[]): UIMessage[] {
  const seenIds = new Set<string>();
  const seenFingerprints = new Set<string>();
  let lastUserMsgID = "";

  return messages.filter((msg) => {
    if (seenIds.has(msg.id)) return false;
    seenIds.add(msg.id);

    if (msg.role === "user") {
      // Track the ID (not text) of the latest user message so we can scope
      // assistant fingerprints to their conversational turn. Using the ID
      // means two user messages with identical text are still treated as
      // distinct turns, preventing false-positive deduplication.
      lastUserMsgID = msg.id;
    }

    if (msg.role === "assistant") {
      // JSON.stringify the parts array to avoid separator-collision false
      // positives: a plain join("|") on ["a|b", "c"] and ["a", "b|c"]
      // produces the same string. JSON encoding each element is unambiguous.
      // Fall back to JSON.stringify(p) for parts that carry neither a text nor
      // a toolCallId (e.g. step-start) so structurally different parts never
      // collapse to the same empty-string fingerprint element.
      const contentFingerprint = JSON.stringify(
        msg.parts.map(
          (p) =>
            ("text" in p && p.text) ||
            ("toolCallId" in p && p.toolCallId) ||
            JSON.stringify(p),
        ),
      );

      if (contentFingerprint !== "[]") {
        // Scope to the preceding user message turn so that identical assistant
        // replies to *different* user prompts are preserved.
        // NOTE: A streaming (in-progress) assistant message has a partial
        // fingerprint that differs from its final form, so it would not be
        // caught by this dedup. This is safe because every caller that invokes
        // resumeStream() first strips the in-progress assistant message —
        // handleReconnect, the wake-resync path, and the hydration-effect path
        // all do this. See useCopilotStream.ts.
        const contextKey = `assistant:${lastUserMsgID}:${contentFingerprint}`;
        if (seenFingerprints.has(contextKey)) return false;
        seenFingerprints.add(contextKey);
      }
    }

    return true;
  });
}
