import type { Expert } from "@/app/api/__generated__/models/expert";

export const RECENT_FACTS_LIMIT = 6;

export const AUTOPILOT_SCOPE = "autopilot";

export const MEMORY_CHAT_PROMPTS = {
  summary: "Give me a summary of everything you know about me",
  forget:
    "I'd like you to forget something from your memory. Ask me what to " +
    "forget, search your memory for matching facts, show me what you find, " +
    "and only delete after I confirm.",
} as const;

export type MemoryChatSeed = keyof typeof MEMORY_CHAT_PROMPTS;

interface MemoryChatHeldKeyArgs {
  /** The signed-in user, who typed it. */
  viewerID: string;
  /** The user an admin is acting as, whose memory it is; null when none. */
  actingAsID: string | null;
  scopeExpertID: string | null;
}

/**
 * Where the memory chat keeps a follow-up it is holding. Every open starts a
 * new session, so one left behind waits for the next chat about the same
 * memory rather than for its own session, which never opens again. It names
 * who typed it and whose memory it is, so it never comes back in another
 * account's chat: not after a sign-out and sign-in on the same tab, and not
 * across an admin acting as a user.
 */
export function memoryChatHeldKey({
  viewerID,
  actingAsID,
  scopeExpertID,
}: MemoryChatHeldKeyArgs) {
  return `memory-chat:${viewerID}:${actingAsID ?? viewerID}:${scopeExpertID ?? "personal"}`;
}

export function getActiveExperts(experts: Expert[] | undefined) {
  return (experts ?? []).filter((expert) => !expert.is_archived);
}

export function getScopeName(
  scopeExpertID: string | null,
  experts: Expert[] | undefined,
) {
  if (!scopeExpertID) return "Otto";
  const expert = experts?.find((e) => e.id === scopeExpertID);
  return expert?.name ?? "this expert";
}

/** Panel identity for a learning record: the Expert, or the personal scope. */
export function learningSheetScope(
  expertId: string | null,
  experts: Expert[] | undefined,
) {
  const expert = expertId
    ? experts?.find((candidate) => candidate.id === expertId)
    : undefined;
  return {
    expertId: expert ? expert.id : null,
    name: expert ? expert.name : "Otto",
    avatarUrl: expert ? expert.avatar_url : null,
    color: expert ? expert.color : null,
  };
}

/** The Soul drawer links here with ?expert=<id>. An id the caller no longer
 *  owns is left to the page's active-expert fallback rather than checked here. */
export function readExpertScopeFromUrl() {
  if (typeof window === "undefined") return null;
  return new URLSearchParams(window.location.search).get("expert") || null;
}
