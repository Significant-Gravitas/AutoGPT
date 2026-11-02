import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";

export function getReviewTitle(review: PendingHumanReviewModel): string {
  return review.instructions || review.agent_name || "Review needed";
}
