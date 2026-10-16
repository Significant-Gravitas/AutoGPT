import type { ChatSessionMetadata } from "@/app/api/__generated__/models/chatSessionMetadata";
import { isValidUUID } from "@/lib/utils";
import type { UIMessage } from "ai";
import { extractDbSequence } from "./helpers/convertChatSessionToUiMessages";

export interface SentFrom {
  sessionId: string;
  expertId: string | null;
  expertName: string | null;
}

type SessionDelegation = Pick<
  ChatSessionMetadata,
  "delegated_by_session_id" | "delegated_by_expert_id"
>;

function readString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

// A delegated, handed-off, sub-session or session-to-session message carries
// the sending session on its own metadata row (see the backend's
// sent_from_metadata helper).
export function getSentFromMetadata(metadata: unknown): SentFrom | null {
  if (!metadata || typeof metadata !== "object") return null;
  const value = metadata as Record<string, unknown>;
  const sessionId = readString(value.from_session_id);
  if (!sessionId || !isValidUUID(sessionId)) return null;
  return {
    sessionId,
    expertId: readString(value.from_expert_id),
    expertName: readString(value.from_expert_name),
  };
}

// Threads opened by a delegation before messages carried provenance only
// record it on the session, which covers the message that opened them.
export function getSessionSentFrom(
  metadata: SessionDelegation | null | undefined,
): SentFrom | null {
  const sessionId = metadata?.delegated_by_session_id;
  if (!sessionId || !isValidUUID(sessionId)) return null;
  return {
    sessionId,
    expertId: metadata?.delegated_by_expert_id ?? null,
    expertName: null,
  };
}

// The row that opened a thread is the one at DB sequence 0. Its hydrated id
// carries that sequence, so the check does not depend on how much history the
// client has retained: a paged, truncated or refetching thread can drop the
// opening row, but it can never present a later row at sequence 0.
export function isSessionOpeningMessage(message: UIMessage): boolean {
  return message.role === "user" && extractDbSequence(message) === 0;
}

export function getSentFromDisplayName(
  sentFrom: SentFrom,
  resolvedExpertName: string | null | undefined,
): string {
  if (sentFrom.expertName) return sentFrom.expertName;
  if (resolvedExpertName) return resolvedExpertName;
  return sentFrom.expertId ? "an expert" : "Otto";
}
