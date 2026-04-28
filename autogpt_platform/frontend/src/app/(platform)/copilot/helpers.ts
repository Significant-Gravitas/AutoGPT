import { getSystemHeaders } from "@/lib/impersonation";
import { getWebSocketToken } from "@/lib/supabase/actions";
import type { UIMessage } from "ai";

import { deleteV2DisconnectSessionStream } from "@/app/api/__generated__/endpoints/chat/chat";
import { TOOL_PART_PREFIX } from "./components/JobStatsBar/constants";

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
 * Resolve the actual dry_run value for a session from the raw API response.
 * Returns true only when the session response is a 200 with metadata.dry_run === true.
 * Returns false for missing/non-200 responses so callers never show a stale
 * preference value when the real session state is unknown.
 */
export function resolveSessionDryRun(queryData: unknown): boolean {
  if (
    queryData == null ||
    typeof queryData !== "object" ||
    !("status" in queryData) ||
    (queryData as { status: unknown }).status !== 200
  )
    return false;
  const d = queryData as { data?: { metadata?: { dry_run?: unknown } } };
  return d.data?.metadata?.dry_run === true;
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

/**
 * Whether the trailing assistant message has at least one part the UI
 * would visibly render: text with non-empty content, reasoning with
 * non-empty content, or any tool part (tool cards render regardless of
 * state). Used to gate the resume-snapshot discard — the replay may stream
 * empty reasoning-start / step-start chunks for minutes before any
 * rendered content (e.g. Perplexity deep research), and we do not want to
 * drop the pre-replay snapshot until the user actually sees something.
 */
export function hasVisibleAssistantContent(messages: UIMessage[]): boolean {
  const last = messages[messages.length - 1];
  if (last?.role !== "assistant") return false;
  return last.parts.some((part) => {
    if (part.type === "text" && part.text.trim().length > 0) return true;
    if (part.type === "reasoning" && part.text.trim().length > 0) return true;
    if (part.type.startsWith(TOOL_PART_PREFIX)) return true;
    return false;
  });
}

/**
 * Surface the latest backend-emitted status message for the trailing assistant
 * message, if that status has not already been invalidated by newer visible
 * parts. Used to show progress during restore/replay before answer text lands.
 */
export function getLatestAssistantStatusMessage(
  messages: UIMessage[],
): string | null {
  const last = messages[messages.length - 1];
  if (last?.role !== "assistant") return null;
  for (let i = last.parts.length - 1; i >= 0; i--) {
    const part = last.parts[i];
    if (part.type === "data-cursor") continue;
    if (part.type === "data-status") {
      const data = (part as { data?: { message?: unknown } }).data;
      return typeof data?.message === "string" ? data.message : null;
    }
    return null;
  }
  return null;
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

const IN_PROGRESS_PART_STATES = new Set([
  "streaming",
  "input-streaming",
  "input-available",
]);

/**
 * True if the message is an assistant message with at least one part that
 * the stream never finalised — i.e. text / reasoning in ``streaming`` or
 * tool parts in ``input-streaming`` / ``input-available``. Used both for
 * the partial-snapshot discard during resume and for zombie-part recovery
 * on session re-entry.
 */
export function hasInProgressAssistantParts(
  message: UIMessage | undefined,
): boolean {
  if (message?.role !== "assistant") return false;
  return message.parts.some((part) => {
    if (!("state" in part) || typeof part.state !== "string") return false;
    return IN_PROGRESS_PART_STATES.has(part.state);
  });
}

const COPILOT_INTERRUPTED_MARKER =
  "[__COPILOT_RETRYABLE_ERROR_a9c2__] Response was interrupted. Resend to try again.";

/**
 * Close the last assistant message when the stream ended without
 * finalising it (backend crash mid-write, the user switched away and the
 * DB snapshot rehydrated with orphaned in-progress parts, etc.). Tool
 * parts in ``input-streaming`` / ``input-available`` flip to
 * ``output-error`` "Interrupted" so their spinners stop; text / reasoning
 * parts in ``streaming`` flip to ``done`` so their typing animation ends
 * but the partial content is preserved. A retryable-error marker is
 * appended so the UI renders a "resend to try again" affordance.
 *
 * Only the last message is touched — earlier messages can't have unclosed
 * parts in a healthy session. Returns the original array when no repair
 * is needed, so callers can cheaply compare references.
 */
export function resolveInterruptedMessage(messages: UIMessage[]): UIMessage[] {
  if (messages.length === 0) return messages;
  const lastIdx = messages.length - 1;
  const last = messages[lastIdx];
  if (!hasInProgressAssistantParts(last)) return messages;

  const resolvedParts = last.parts.map((part) => {
    if (!("state" in part) || typeof part.state !== "string") return part;
    if (part.state === "input-streaming" || part.state === "input-available") {
      return {
        ...part,
        state: "output-error" as const,
        errorText: "Interrupted",
      };
    }
    if (part.state === "streaming") {
      return { ...part, state: "done" as const };
    }
    return part;
  });

  return [
    ...messages.slice(0, lastIdx),
    {
      ...last,
      parts: [
        ...resolvedParts,
        { type: "text" as const, text: COPILOT_INTERRUPTED_MARKER },
      ],
    },
  ];
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
 * Fire-and-forget: tell the backend to release XREAD listeners for a session.
 *
 * Called on session switch so the backend doesn't wait for its 5-10 s timeout
 * before cleaning up. Failures are silently ignored — the backend will
 * eventually clean up on its own.
 */
export function disconnectSessionStream(sessionId: string): void {
  deleteV2DisconnectSessionStream(sessionId).catch(() => {});
}

/**
 * Decide whether a reconnect request must be coalesced onto the debounce
 * window boundary, rather than firing immediately.
 *
 * Returns the remaining milliseconds until the window closes (so the caller
 * can schedule a `setTimeout` for that delay) when the previous resume
 * happened inside the window, or `null` to let the reconnect proceed now.
 *
 * `lastResumeAt === 0` signals "no reconnect has fired yet in this session"
 * — the first reconnect always passes through regardless of `now`.
 */
export function shouldDebounceReconnect(
  lastResumeAt: number,
  now: number,
  windowMs: number,
): number | null {
  if (lastResumeAt <= 0) return null;
  const sinceLastResume = now - lastResumeAt;
  if (sinceLastResume >= windowMs) return null;
  return windowMs - sinceLastResume;
}

/**
 * Deduplicate messages by ID and by consecutive content fingerprint.
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
