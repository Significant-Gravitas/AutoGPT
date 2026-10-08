import { getSystemHeaders } from "@/lib/impersonation";
import { getWebSocketToken } from "@/lib/auth/actions";
import type { ChatStatus, UIMessage } from "ai";

import { deleteV2DisconnectSessionStream } from "@/app/api/__generated__/endpoints/chat/chat";
import { TOOL_PART_PREFIX } from "./components/JobStatsBar/constants";
import { parseSpecialMarkers } from "./helpers/messageMarkers";

export const ORIGINAL_TITLE = "AutoGPT";

/**
 * Title/body/icon for the OS-level notification fired when a copilot session
 * completes. Kept in sync with the same copy hardcoded in `public/push-sw.js`
 * (NOTIFICATION_MAP.copilot_completion.session_completed) — the SW file is
 * plain JS served from /public and can't import from this module, so the two
 * sources are matched by test rather than by reference.
 */
export const COPILOT_COMPLETION_NOTIFICATION = {
  title: "AutoGPT",
  body: "Task completed",
  icon: "/notification-icon-192.png",
} as const;

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
    ? `(${count}) New activity - ${ORIGINAL_TITLE}`
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
  return getActiveBackendTurnId(result) !== null;
}

/** The turn a refetchSession result reports running; "" when the backend
 *  reports a stream without naming its turn, null when none runs. */
export function getActiveBackendTurnId(result: {
  data?: unknown;
}): string | null {
  const d = result.data as
    | { status?: unknown; data?: { active_stream?: { turn_id?: unknown } } }
    | null
    | undefined;
  if (d?.status !== 200) return null;
  const active = d.data?.active_stream;
  if (!active) return null;
  return typeof active.turn_id === "string" ? active.turn_id : "";
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
  status?: ChatStatus;
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
  status,
}: SuppressDuplicateArgs): SuppressReason {
  if (isReconnectScheduled) return "reconnecting";

  const lastMessage = messages.at(-1);
  const hasRetryableError =
    status === "ready" &&
    lastMessage?.role === "assistant" &&
    lastMessage.parts.some(
      (part) =>
        part.type === "text" &&
        parseSpecialMarkers(part.text).markerType === "retryable_error",
    );
  if (status === "error" || hasRetryableError) return null;

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
 * Drop messages whose id an earlier message already holds, keeping the first.
 *
 * Content is never compared: the turn stream hands the parser each entry
 * once, so a replay cannot add a copy of a message under a new id.
 */
export function deduplicateMessages(messages: UIMessage[]): UIMessage[] {
  const seenIds = new Set<string>();
  return messages.filter((msg) => {
    if (seenIds.has(msg.id)) return false;
    seenIds.add(msg.id);
    return true;
  });
}

/**
 * True when the server reports it moved a turn to a different execution
 * engine. Nothing displays the engine and nothing can request one — the only
 * consumer widens its post-finish refetch window, because a switch takes
 * longer to settle. The named engine is deliberately not returned.
 */
export function isEngineSwitchPart(dataPart: {
  type: string;
  data?: unknown;
}): boolean {
  if (dataPart.type !== "data-mode-changed") return false;
  const mode = (dataPart.data as { mode?: string } | undefined)?.mode;
  return mode === "extended_thinking" || mode === "fast";
}
