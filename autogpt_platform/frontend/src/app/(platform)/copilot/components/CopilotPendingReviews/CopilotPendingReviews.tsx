"use client";

import { useCallback } from "react";
import {
  COPILOT_GATE_NODE_PREFIX,
  PendingReviewsList,
} from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { useCopilotChatActions } from "../CopilotChatActionsProvider/useCopilotChatActions";
import { okData } from "@/app/api/helpers";
import { ApprovalQueue } from "../ApprovalQueue/ApprovalQueue";
import { isGateReview, toApprovalItem } from "../ApprovalQueue/helpers";
import { useCopilotPendingReviews } from "./useCopilotPendingReviews";

type Props =
  | { graphExecId: string; graphId?: string }
  | { chatSessionId: string; pollWhileEmpty?: boolean; refetchKey?: number };

/**
 * Renders the chat's pending reviews, or those of an agent run it started:
 * held AutoPilot calls in one "Waiting for you" queue, oldest first, and
 * every block, MCP or run review in one consolidated list.
 */
export function CopilotPendingReviews(props: Props) {
  const { onSend, onBackendTurn } = useCopilotChatActions();
  const graphExecId = "graphExecId" in props ? props.graphExecId : "";
  const { pendingReviews, refetch } = useCopilotPendingReviews(props);

  const heldCalls = pendingReviews
    .filter(isGateReview)
    .sort((a, b) => +new Date(a.created_at) - +new Date(b.created_at))
    .map(toApprovalItem);
  const otherReviews = pendingReviews.filter(
    (r) => !r.node_exec_id.startsWith(COPILOT_GATE_NODE_PREFIX),
  );

  async function handleHeldCallAnswered() {
    await refetch();
    onBackendTurn?.();
  }

  const handleReviewComplete = useCallback(async () => {
    // Brief delay for the server to propagate the approval
    await new Promise((resolve) => setTimeout(resolve, 500));
    const result = await refetch();
    const remaining = (okData(result.data) || []).filter(
      (r) => !r.node_exec_id.startsWith(COPILOT_GATE_NODE_PREFIX),
    );

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

  return (
    <div className="flex flex-col gap-2 py-2 empty:hidden">
      <ApprovalQueue items={heldCalls} onAnswered={handleHeldCallAnswered} />
      {otherReviews.length > 0 && (
        <PendingReviewsList
          reviews={otherReviews}
          onReviewComplete={handleReviewComplete}
        />
      )}
    </div>
  );
}
