import { useState } from "react";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useNeedsAttention } from "@/hooks/useNeedsAttention";
import { useProcessReviews } from "@/hooks/useProcessReviews";

export function useNeedsAttentionList() {
  const { reviews, countLabel, isLoading, isError, refetch } =
    useNeedsAttention();
  const { processReviews } = useProcessReviews();
  const { toast } = useToast();
  // Tracked per row rather than reusing the mutation's shared isPending, so
  // acting on one review doesn't disable every other row.
  const [pendingNodeExecId, setPendingNodeExecId] = useState<string | null>(
    null,
  );

  async function decide(
    review: PendingHumanReviewModel,
    approved: boolean,
  ): Promise<void> {
    const verb = approved ? "approve" : "decline";
    setPendingNodeExecId(review.node_exec_id);
    try {
      await processReviews(
        [
          {
            node_exec_id: review.node_exec_id,
            approved,
            ...(approved ? {} : { message: "Declined from home" }),
            auto_approve_future: false,
          },
        ],
        [review.graph_exec_id],
      );
      toast({ title: approved ? "Approved" : "Declined" });
    } catch (error) {
      toast({
        title: `Failed to ${verb} review`,
        description:
          error instanceof Error ? error.message : "An error occurred",
        variant: "destructive",
      });
    } finally {
      setPendingNodeExecId(null);
    }
  }

  function approve(review: PendingHumanReviewModel) {
    return decide(review, true);
  }

  function decline(review: PendingHumanReviewModel) {
    return decide(review, false);
  }

  return {
    reviews,
    countLabel,
    isLoading,
    isError,
    refetch,
    pendingNodeExecId,
    approve,
    decline,
  };
}
