import type { ExpertRecommendations } from "@/app/api/__generated__/models/expertRecommendations";
import type { RecommendedExpert } from "@/app/api/__generated__/models/recommendedExpert";

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
// Long enough to see the last card flip to Hired before the step moves on.
export const HIRE_ALL_ADVANCE_DELAY_MS = 500;

export type ContinueVia = "auto" | "continue" | "later";

export const SKIP_ACTION = {
  label: "I'll add members later",
  via: "later",
} as const;

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

export function raiseNote(role: string | null | undefined): string | null {
  if (!role) return null;
  return `Nobody on the roster covers ${role} yet — you can create your own ${role} once you're set up.`;
}

interface ActionArgs {
  experts: RecommendedExpert[];
  hiredTemplateIds: string[];
  failedTemplateIds: string[];
  excludedTemplateIds: string[];
}

// The primary hires every unhired expert the user left in, and moves on once
// nobody is unhired; with everyone left out, the skip is the only action. The
// secondary always moves on without hiring more.
export function hireStepActions({
  experts,
  hiredTemplateIds,
  failedTemplateIds,
  excludedTemplateIds,
}: ActionArgs) {
  const unhired = experts.filter(
    (expert) => !hiredTemplateIds.includes(expert.template_id),
  );
  const batch = unhired.filter(
    (expert) => !excludedTemplateIds.includes(expert.template_id),
  );
  const hiredCount = experts.length - unhired.length;
  const isRetry =
    batch.length > 0 &&
    batch.every((expert) => failedTemplateIds.includes(expert.template_id));
  const skip =
    hiredCount > 0
      ? ({ label: "Continue", via: "continue" } as const)
      : SKIP_ACTION;

  if (unhired.length === 0) {
    return { primaryLabel: "Continue", batch, isRetry, skip: null };
  }
  return {
    primaryLabel: isRetry
      ? "Try again"
      : hireLabel(batch, unhired.length, hiredCount),
    batch,
    isRetry,
    skip,
  };
}

export function hireFailureToast(
  failed: RecommendedExpert[],
  batchSize: number,
  anyHired: boolean,
) {
  if (!anyHired && batchSize > 1 && failed.length === batchSize) {
    return {
      title: "Couldn't hire your team",
      description: "Something went wrong. Please try again.",
    };
  }
  return {
    title: `Couldn't hire ${failedNames(failed)}`,
    description: anyHired
      ? "Something went wrong. Try again, or continue without them."
      : "Something went wrong. Please try again.",
  };
}

function hireLabel(
  batch: RecommendedExpert[],
  unhiredCount: number,
  hiredCount: number,
) {
  if (batch.length === 0) return null;
  if (batch.length === 1) {
    return `Hire ${batch[0].name}${hiredCount > 0 ? " too" : ""}`;
  }
  if (batch.length < unhiredCount) return `Hire ${batch.length}`;
  return hiredCount > 0
    ? `Hire the other ${batch.length}`
    : `Hire all ${batch.length}`;
}

function failedNames(failed: RecommendedExpert[]) {
  const names = failed.map((expert) => expert.name);
  if (names.length > 3) return `${names.length} experts`;
  if (names.length === 1) return names[0];
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}
