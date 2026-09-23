"use client";

import { useCallback } from "react";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { useCopilotChatActions } from "../CopilotChatActionsProvider/useCopilotChatActions";
import {
  usePendingReviewsForExecution,
  usePendingReviewsForSession,
} from "@/hooks/usePendingReviews";
import { okData } from "@/app/api/helpers";

type Props = { graphExecId: string } | { sessionId: string };

const POLL = { refetchInterval: 2000 };

/**
 * Renders a single consolidated PendingReviewsList for the chat's own reviews
 * (run_capability, MCP, spend approval) or for an agent run the chat started
 * — mirrors the non-copilot review page behavior.
 */
export function CopilotPendingReviews(props: Props) {
  const { onSend } = useCopilotChatActions();
  const graphExecId = "graphExecId" in props ? props.graphExecId : "";
  const sessionId = "sessionId" in props ? props.sessionId : "";
  const forRun = usePendingReviewsForExecution(graphExecId, {
    ...POLL,
    enabled: !!graphExecId,
  });
  const forChat = usePendingReviewsForSession(sessionId, {
    ...POLL,
    enabled: !!sessionId,
  });
  const { pendingReviews, refetch } = graphExecId ? forRun : forChat;

  const handleReviewComplete = useCallback(async () => {
    // Brief delay for the server to propagate the approval
    await new Promise((resolve) => setTimeout(resolve, 500));
    const result = await refetch();
    const remaining = okData(result.data) || [];

    if (remaining.length > 0) return;

    // Graph executions auto-resume after approval; chat reviews need resume_capability.
    if (graphExecId) {
      onSend(
        `All pending reviews have been processed. ` +
          `The agent execution will resume automatically for approved reviews. ` +
          `Use view_agent_output with execution_id="${graphExecId}" to check the result.`,
      );
    } else {
      onSend(
        `All pending reviews have been processed. ` +
          `For any approved reviews, call resume_capability with the corresponding review_id to execute them. ` +
          `For rejected reviews, no further action is needed.`,
      );
    }
  }, [refetch, onSend, graphExecId]);

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
