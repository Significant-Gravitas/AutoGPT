"use client";

import { useCallback } from "react";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { useCopilotChatActions } from "../CopilotChatActionsProvider/useCopilotChatActions";
import { okData } from "@/app/api/helpers";
import { useCopilotPendingReviews } from "./useCopilotPendingReviews";

interface Props {
  graphExecId: string;
  graphId?: string;
}

/**
 * Renders a single consolidated PendingReviewsList for all pending copilot
 * reviews in a session — mirrors the non-copilot review page behavior.
 * Works for both run_capability (synthetic copilot-session-*) and run_agent (real graph exec) reviews.
 */
export function CopilotPendingReviews({ graphExecId, graphId }: Props) {
  const { onSend } = useCopilotChatActions();
  const { pendingReviews, refetch } = useCopilotPendingReviews({
    graphExecId,
    graphId,
  });

  // Graph executions auto-resume after approval; capability reviews need resume_capability.
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
      // Gate approvals are consumed by re-issuing the original tool call, not
      // by resume_capability — that is only for capability reviews.
      onSend(
        `All pending reviews have been processed. ` +
          `For an approved block or MCP review, call resume_capability with the ` +
          `corresponding review_id. For any other approved action, retry the ` +
          `tool call you were blocked on, with the same arguments. ` +
          `For rejected reviews, do not retry — tell me what you could not do.`,
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
