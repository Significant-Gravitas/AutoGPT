import type { ExpertRecommendations } from "@/app/api/__generated__/models/expertRecommendations";

export const HIRE_TITLE_LLM = "Based on what you told me, here's who I'd hire";
export const HIRE_TITLE_FALLBACK =
  "Based on your role, here's who I'd hire first";
export const HIRE_TITLE_EMPTY = "Here's my read on your team";
// No experts and no diagnosis to show: the title must not promise a read
// that never renders.
export const HIRE_TITLE_EMPTY_NO_READ =
  "Nothing to hire yet — you can add members later";
export const HIRE_TITLE_PENDING = "Thinking about who you'd need…";

export const RECOMMENDATIONS_POLL_MS = 2_500;
// A job the backend never finished (process restart mid-run) must not strand
// the user on this step: past this ceiling the step shows what it has, or the
// way out, rather than a skeleton forever.
export const RECOMMENDATIONS_MAX_WAIT_MS = 20_000;

export function hireTitle(
  team: ExpertRecommendations | null,
  isPending: boolean,
): string {
  if (isPending) return HIRE_TITLE_PENDING;
  if ((team?.experts?.length ?? 0) === 0) {
    return team?.diagnosis ? HIRE_TITLE_EMPTY : HIRE_TITLE_EMPTY_NO_READ;
  }
  return team?.source === "llm" ? HIRE_TITLE_LLM : HIRE_TITLE_FALLBACK;
}

// The diagnosis is the "read" the empty-state title promises; with cards on
// screen the cards themselves are the read, so it stays out of the way.
export function hireDiagnosis(
  team: ExpertRecommendations | null,
  isPending: boolean,
): string | null {
  if (isPending || (team?.experts?.length ?? 0) > 0) return null;
  return team?.diagnosis?.trim() || null;
}

export function continueLabel(hiredCount: number): string {
  return hiredCount > 0 ? "Next" : "I'll add members later";
}

export function raiseNote(role: string | null | undefined): string | null {
  if (!role) return null;
  return `Nobody on the roster covers ${role} yet — you can raise your own ${role} once you're set up.`;
}
