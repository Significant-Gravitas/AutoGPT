import type { Expert } from "@/app/api/__generated__/models/expert";
import { formatDistanceToNow } from "date-fns";

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

export function getActiveExperts(experts: Expert[] | undefined) {
  return (experts ?? []).filter((expert) => !expert.is_archived);
}

export function getScopeName(
  scopeExpertID: string | null,
  experts: Expert[] | undefined,
) {
  if (!scopeExpertID) return "AutoPilot";
  const expert = experts?.find((e) => e.id === scopeExpertID);
  return expert?.name ?? "this expert";
}

export function formatWhen(createdAt: string | null | undefined) {
  if (!createdAt) return "";
  const parsed = new Date(createdAt);
  if (Number.isNaN(parsed.getTime())) return "";
  return formatDistanceToNow(parsed, { addSuffix: true });
}
