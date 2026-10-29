import type {
  listMyExpertMemoryFacts,
  listMyMemoryFacts,
} from "@/app/api/__generated__/endpoints/memory/memory";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { formatDistanceToNow } from "date-fns";

export const RECENT_FACTS_LIMIT = 6;

/** One request covers the preview and, for most people, "Show all" too. */
export const FACTS_PAGE_SIZE = 50;

type FactsPage =
  | Awaited<ReturnType<typeof listMyMemoryFacts>>
  | Awaited<ReturnType<typeof listMyExpertMemoryFacts>>;

export function getLoadedFacts(pages: FactsPage[] | undefined) {
  return (pages ?? []).flatMap((page) =>
    page.status === 200 ? page.data.items : [],
  );
}

export function getFactsTotal(pages: FactsPage[] | undefined) {
  const first = pages?.[0];
  return first?.status === 200 ? first.data.total : null;
}

export function getNextFactsCursor(page: FactsPage) {
  return page.status === 200 ? (page.data.next_cursor ?? undefined) : undefined;
}

export const AUTOPILOT_SCOPE = "autopilot";

export const MEMORY_CHAT_PROMPTS = {
  summary: "Give me a summary of everything you know about me",
  forget:
    "I'd like you to forget something from your memory. Ask me what to " +
    "forget, search your memory for matching facts, show me what you find, " +
    "and only delete after I confirm.",
} as const;

export type MemoryChatSeed = keyof typeof MEMORY_CHAT_PROMPTS;

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

export function formatWhen(createdAt: string | null | undefined) {
  if (!createdAt) return "";
  const parsed = new Date(createdAt);
  if (Number.isNaN(parsed.getTime())) return "";
  return formatDistanceToNow(parsed, { addSuffix: true });
}

/** The Soul drawer links here with ?expert=<id>. An id the caller no longer
 *  owns is left to the page's active-expert fallback rather than checked here. */
export function readExpertScopeFromUrl() {
  if (typeof window === "undefined") return null;
  return new URLSearchParams(window.location.search).get("expert") || null;
}
