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
 * Works for both run_block (synthetic copilot-session-*) and run_agent (real graph exec) reviews.
 */
export function CopilotPendingReviews({ graphExecId }: Props) {
  const { onSend } = useCopilotChatActions();
  const { pendingReviews, refetch } = usePendingReviewsForExecution(
    graphExecId,
    { enabled: !!graphExecId, refetchInterval: 2000 },
  );

  // Graph executions auto-resume after approval; block reviews need continue_run_block.
  const isGraphExecution = !graphExecId.startsWith("copilot-session-");

  const handleReviewComplete = useCallback(async () => {
    // Brief delay for the server to propagate the approval
    await new Promise((resolve) => setTimeout(resolve, 500));
    const result = await refetch();
    const remaining = okData(result.data) || [];

    if (remaining.length > 0) return;

    if (isGraphExecution) {
      onSend(
        `All pending reviews have been processed. ` +
          `The agent execution will resume automatically for approved reviews. ` +
          `Use view_agent_output with execution_id="${graphExecId}" to check the result.`,
      );
    } else {
      onSend(
        `All pending reviews have been processed. ` +
          `For any approved reviews, call continue_run_block with the corresponding review_id to execute them. ` +
          `For rejected reviews, no further action is needed.`,
      );
    }
  }, [refetch, onSend, isGraphExecution, graphExecId]);

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
