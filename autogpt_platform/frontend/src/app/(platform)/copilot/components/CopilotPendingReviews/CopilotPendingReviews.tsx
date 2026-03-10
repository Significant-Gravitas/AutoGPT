"use client";

import { useCallback } from "react";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { useCopilotChatActions } from "../CopilotChatActionsProvider/useCopilotChatActions";
import { usePendingReviewsForExecution } from "@/hooks/usePendingReviews";
import { okData } from "@/app/api/helpers";

interface Props {
  graphExecId: string;
}

/**
 * Renders a single consolidated PendingReviewsList for all pending copilot
 * reviews in a session — mirrors the non-copilot review page behavior.
 */
export function CopilotPendingReviews({ graphExecId }: Props) {
  const { onSend } = useCopilotChatActions();
  const { pendingReviews, refetch } = usePendingReviewsForExecution(
    graphExecId,
    { enabled: !!graphExecId, refetchInterval: 2000 },
  );

  const handleReviewComplete = useCallback(async () => {
    // Brief delay for the server to propagate the approval
    await new Promise((resolve) => setTimeout(resolve, 500));
    const result = await refetch();
    const remaining = okData(result.data) || [];

    if (remaining.length > 0) return;

    onSend(
      `All pending reviews have been processed. ` +
        `For any approved reviews, call continue_run_block with the corresponding review_id to execute them. ` +
        `For rejected reviews, no further action is needed.`,
    );
  }, [refetch, onSend]);

  if (pendingReviews.length === 0) return null;

  return (
    <div className="py-2">
      <PendingReviewsList
        reviews={pendingReviews}
        onReviewComplete={handleReviewComplete}
      />
    </div>
  );
}
