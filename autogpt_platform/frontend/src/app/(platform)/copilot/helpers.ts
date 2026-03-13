import type { GetV2ListSessionsParams } from "@/app/api/__generated__/models/getV2ListSessionsParams";
import {
  ChatSessionStartType,
  type ChatSessionStartType as ChatSessionStartTypeValue,
} from "@/app/api/__generated__/models/chatSessionStartType";
import type { UIMessage } from "ai";

export const COPILOT_SESSION_LIST_LIMIT = 50;

export function getSessionListParams(
  includeNonManual: boolean,
): GetV2ListSessionsParams {
  return {
    limit: COPILOT_SESSION_LIST_LIMIT,
    with_auto: includeNonManual,
  };
}

export function isNonManualSessionStartType(
  startType: ChatSessionStartTypeValue | null | undefined,
): boolean {
  return startType != null && startType !== ChatSessionStartType.MANUAL;
}

export function getSessionStartTypeLabel(
  startType: ChatSessionStartTypeValue,
): string | null {
  switch (startType) {
    case ChatSessionStartType.AUTOPILOT_NIGHTLY:
      return "Nightly";
    case ChatSessionStartType.AUTOPILOT_CALLBACK:
      return "Callback";
    case ChatSessionStartType.AUTOPILOT_INVITE_CTA:
      return "Invite CTA";
    default:
      return null;
  }
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
