import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useNeedsAttention } from "@/hooks/useNeedsAttention";
import { useProcessReviews } from "@/hooks/useProcessReviews";

export function useNeedsAttentionList() {
  const { reviews, count, isLoading, isError, refetch } = useNeedsAttention();
  const { processReviews, isProcessing } = useProcessReviews();
  const { toast } = useToast();

  async function approve(review: PendingHumanReviewModel) {
    try {
      await processReviews(
        [
          {
            node_exec_id: review.node_exec_id,
            approved: true,
            auto_approve_future: false,
          },
        ],
        [review.graph_exec_id],
      );
    } catch (error) {
      toast({
        title: "Failed to approve review",
        description:
          error instanceof Error ? error.message : "An error occurred",
        variant: "destructive",
      });
    }
  }

  async function skip(review: PendingHumanReviewModel) {
    try {
      await processReviews(
        [
          {
            node_exec_id: review.node_exec_id,
            approved: false,
            message: "Skipped from home",
            auto_approve_future: false,
          },
        ],
        [review.graph_exec_id],
      );
    } catch (error) {
      toast({
        title: "Failed to skip review",
        description:
          error instanceof Error ? error.message : "An error occurred",
        variant: "destructive",
      });
    }
  }

  return {
    reviews,
    count,
    isLoading,
    isError,
    refetch,
    isProcessing,
    approve,
    skip,
  };
}
