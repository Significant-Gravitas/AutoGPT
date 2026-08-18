import type { Expert } from "@/app/api/__generated__/models/expert";
import { formatDistanceToNow } from "date-fns";

export const RECENT_FACTS_LIMIT = 6;

export const AUTOPILOT_SCOPE = "autopilot";

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
