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
  // A set rather than a single slot: morning triage taps several rows in
  // quick succession, and one slot would unlock row A the moment row B
  // started, making A double-submittable while its POST was still in flight.
  const [pendingNodeExecIds, setPendingNodeExecIds] = useState<Set<string>>(
    new Set(),
  );

  function setPending(nodeExecId: string, isPending: boolean) {
    setPendingNodeExecIds((current) => {
      const next = new Set(current);
      if (isPending) next.add(nodeExecId);
      else next.delete(nodeExecId);
      return next;
    });
  }

  async function decide(
    review: PendingHumanReviewModel,
    approved: boolean,
  ): Promise<void> {
    const verb = approved ? "approve" : "decline";
    setPending(review.node_exec_id, true);
    try {
      const res = await processReviews(
        [
          {
            node_exec_id: review.node_exec_id,
            approved,
            // No message: this surface has no field to write one in, and a
            // canned English string would reach the agent's context and the
            // audit trail as if the user had typed it.
            auto_approve_future: false,
          },
        ],
        [review.graph_exec_id],
      );

      // The mutation resolves rather than throws on a non-200, and a 200 can
      // still carry failed_count > 0 (review already processed, node
      // execution gone). Reporting either as success would leave the row
      // reappearing on the next refetch with nothing explaining why.
      if (res.status !== 200) {
        throw new Error("Unexpected response from server");
      }
      if (res.data.failed_count > 0) {
        throw new Error(res.data.error || "The review could not be processed.");
      }

      toast({ title: approved ? "Approved" : "Declined" });
    } catch (error) {
      toast({
        title: `Failed to ${verb} review`,
        description:
          error instanceof Error ? error.message : "An error occurred",
        variant: "destructive",
      });
    } finally {
      setPending(review.node_exec_id, false);
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
    pendingNodeExecIds,
    approve,
    decline,
  };
}
