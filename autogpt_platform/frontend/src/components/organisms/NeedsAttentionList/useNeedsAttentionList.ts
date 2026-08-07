import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { useNeedsAttention } from "@/hooks/useNeedsAttention";
import { useProcessReviews } from "@/hooks/useProcessReviews";

export function useNeedsAttentionList() {
  const { reviews, count, isLoading } = useNeedsAttention();
  const { processReviews, isProcessing } = useProcessReviews();

  function approve(review: PendingHumanReviewModel) {
    return processReviews(
      [
        {
          node_exec_id: review.node_exec_id,
          approved: true,
          auto_approve_future: false,
        },
      ],
      review.graph_exec_id,
    );
  }

  function skip(review: PendingHumanReviewModel) {
    return processReviews(
      [
        {
          node_exec_id: review.node_exec_id,
          approved: false,
          message: "Skipped from home",
          auto_approve_future: false,
        },
      ],
      review.graph_exec_id,
    );
  }

  return { reviews, count, isLoading, isProcessing, approve, skip };
}
