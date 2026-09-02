import { isValidUUID } from "@/lib/utils";
import type { UIMessage } from "ai";

const EXPERT_KICKOFF_KIND = "expert_kickoff";
const KICKOFF_STORAGE_PREFIX = "expert-kickoff-status:";
const LEGACY_MARKER_PATTERN =
  /^\[\[EXPERT_KICKOFF:([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\]\](?:\n\n)?/i;
const PENDING_TTL_MS = 2 * 60 * 1000;

const KICKOFF_PROMPT =
  "You were just hired. Introduce yourself in 2-3 sentences in your voice. " +
  "If you have installed workflows, state the day-one job they support and " +
  "start the first bundled workflow with run_agent. If no workflow is " +
  "installed, explain the outcomes you can help with and ask which one to " +
  "start. If required access or credentials are missing, ask for exactly the " +
  "one connection the next job needs and why. Never pretend a run succeeded.";

export interface ExpertKickoffMetadata {
  kind: typeof EXPERT_KICKOFF_KIND;
  expertId: string;
  attemptToken?: KickoffAttemptToken;
}

export interface ExpertKickoffSend {
  text: string;
  metadata: ExpertKickoffMetadata;
}

export type KickoffStatus = "idle" | "pending" | "done";
export type KickoffAttemptToken = string;

export function buildKickoffMessage(
  expertId: string,
  attemptToken?: KickoffAttemptToken,
): ExpertKickoffSend {
  return {
    text: KICKOFF_PROMPT,
    metadata: {
      kind: EXPERT_KICKOFF_KIND,
      expertId,
      ...(attemptToken ? { attemptToken } : {}),
    },
  };
}

export function getKickoffExpertIdFromMetadata(
  metadata: unknown,
): string | null {
  if (!metadata || typeof metadata !== "object") return null;
  const value = metadata as Record<string, unknown>;
  if (value.kind !== EXPERT_KICKOFF_KIND) return null;
  const expertId = value.expertId ?? value.expert_id;
  return typeof expertId === "string" && isValidUUID(expertId)
    ? expertId
    : null;
}

export function getKickoffAttemptTokenFromMetadata(
  metadata: unknown,
): KickoffAttemptToken | null {
  if (!metadata || typeof metadata !== "object") return null;
  const value = metadata as Record<string, unknown>;
  if (value.kind !== EXPERT_KICKOFF_KIND) return null;
  const attemptToken = value.attemptToken;
  return typeof attemptToken === "string" &&
    attemptToken.length > 0 &&
    attemptToken.length <= 128
    ? attemptToken
    : null;
}

export function parseLegacyKickoffExpertId(text: string): string | null {
  const match = text.match(LEGACY_MARKER_PATTERN);
  return match?.[1] ?? null;
}

export function isKickoffText(text: string): boolean {
  return parseLegacyKickoffExpertId(text) !== null;
}

export function stripLegacyKickoffMarker(text: string): string {
  return text.replace(LEGACY_MARKER_PATTERN, "");
}

function firstTextPart(message: UIMessage): string | null {
  const part = message.parts.find(
    (candidate): candidate is Extract<typeof candidate, { type: "text" }> =>
      candidate.type === "text",
  );
  return part?.text ?? null;
}

export function getKickoffExpertId(message: UIMessage): string | null {
  if (message.role !== "user") return null;
  const metadataExpertId = getKickoffExpertIdFromMetadata(message.metadata);
  if (metadataExpertId) return metadataExpertId;
  const text = firstTextPart(message);
  return text ? parseLegacyKickoffExpertId(text) : null;
}

export function getKickoffAttemptToken(
  message: UIMessage,
): KickoffAttemptToken | null {
  if (message.role !== "user") return null;
  return getKickoffAttemptTokenFromMetadata(message.metadata);
}

export function isKickoffMessage(message: UIMessage): boolean {
  return getKickoffExpertId(message) !== null;
}

export function shouldClearKickoffParam(
  isExpertsEnabled: boolean,
  hasExpertsSettled: boolean,
  expertId: string | null,
): boolean {
  return !isExpertsEnabled || (hasExpertsSettled && expertId === null);
}

// The kickoff prompt reads in the thread like any other opening message, so a
// freshly raised expert answers something visible rather than thin air. Older
// threads carry an inline marker that was never meant to be read.
export function revealKickoffMessages<T extends UIMessage>(messages: T[]): T[] {
  return messages.map((message) => {
    if (!isKickoffMessage(message)) return message;
    return {
      ...message,
      parts: message.parts.map((part) =>
        part.type === "text"
          ? { ...part, text: stripLegacyKickoffMarker(part.text) }
          : part,
      ),
    };
  });
}

export function kickoffStorageKey(userId: string, expertId: string): string {
  return `${KICKOFF_STORAGE_PREFIX}${userId}:${expertId}`;
}

function readKickoffStorage(userId: string, expertId: string): string | null {
  return window.localStorage.getItem(kickoffStorageKey(userId, expertId));
}

export function getKickoffStatus(
  userId: string,
  expertId: string,
): KickoffStatus {
  if (typeof window === "undefined") return "idle";
  try {
    const value = readKickoffStorage(userId, expertId);
    if (value === null) return "idle";
    if (value === "done" || value === "1" || value.startsWith("done:")) {
      return "done";
    }
    if (!value.startsWith("pending:")) return "idle";
    const startedAt = Number(value.split(":", 2)[1]);
    return Number.isFinite(startedAt) && Date.now() - startedAt < PENDING_TTL_MS
      ? "pending"
      : "idle";
  } catch {
    return "idle";
  }
}

export function markKickoffPending(
  userId: string,
  expertId: string,
): KickoffAttemptToken {
  const attemptToken = `${Date.now()}:${crypto.randomUUID()}`;
  if (typeof window === "undefined") return attemptToken;
  try {
    window.localStorage.setItem(
      kickoffStorageKey(userId, expertId),
      `pending:${attemptToken}`,
    );
  } catch {
    return attemptToken;
  }
  return attemptToken;
}

export function markKickoffDone(
  userId: string,
  expertId: string,
  attemptToken: KickoffAttemptToken,
): boolean {
  if (typeof window === "undefined") return false;
  try {
    const current = readKickoffStorage(userId, expertId);
    const completed = `done:${attemptToken}`;
    if (current === completed) return true;
    if (current !== `pending:${attemptToken}`) return false;
    window.localStorage.setItem(kickoffStorageKey(userId, expertId), completed);
    return true;
  } catch {
    return false;
  }
}

export function clearKickoffPending(
  userId: string,
  expertId: string,
  attemptToken: KickoffAttemptToken,
): void {
  if (typeof window === "undefined") return;
  try {
    const value = readKickoffStorage(userId, expertId);
    if (value === `pending:${attemptToken}`) {
      window.localStorage.removeItem(kickoffStorageKey(userId, expertId));
    }
  } catch {
    return;
  }
}

export async function withKickoffLock<T>(
  userId: string,
  expertId: string,
  action: () => Promise<T>,
): Promise<T | undefined> {
  if (typeof navigator === "undefined" || !navigator.locks) return action();
  return navigator.locks.request(kickoffStorageKey(userId, expertId), action);
}
