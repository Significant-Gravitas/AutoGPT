import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";

export type SpendSummary = NonNullable<ReturnType<typeof spendSummary>>;

// Reads the payload `spend_approval.py::_review_payload` writes. Every review
// in a group is the same expert in the same window; runs parked later saw a
// higher spend, so the largest is the current one.
export function spendSummary(reviews: PendingHumanReviewModel[]) {
  const first = reviews[0];
  if (!first) return null;
  const payload = asRecord(first.payload);
  const threshold = payload.threshold_credits;
  const spent = Math.max(...reviews.map(spentCredits));
  if (typeof threshold !== "number" || !Number.isFinite(spent)) return null;

  return {
    expertName:
      typeof payload.expert === "string" && payload.expert
        ? payload.expert
        : (first.expert_name ?? "Your expert"),
    avatarUrl: first.expert_avatar_url ?? null,
    spent,
    threshold,
    period: payload.window === "week" ? "this week" : "today",
    waiting: reviews.length,
  };
}

export function spendSubtitle(summary: SpendSummary) {
  const past = `That's past your ${summary.threshold} check-in.`;
  if (summary.waiting === 1) return `${past} Keep going?`;
  return `${past} ${summary.waiting} tasks are ready to go.`;
}

function spentCredits(review: PendingHumanReviewModel) {
  const spent = asRecord(review.payload).spent_credits;
  return typeof spent === "number" ? spent : Number.NaN;
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}
