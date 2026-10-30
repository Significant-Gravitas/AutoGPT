"use client";

import {
  COPILOT_GATE_NODE_PREFIX,
  PendingReviewsList,
} from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { useCopilotChatActions } from "../CopilotChatActionsProvider/useCopilotChatActions";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import type { ExpertIdentity } from "../../useExpertMap";
import type { RunReviewTarget } from "../ChatMessagesContainer/helpers";
import { ApprovalQueue } from "../ApprovalQueue/ApprovalQueue";
import { isGateReview, toApprovalItem } from "../ApprovalQueue/helpers";
import { useCopilotPendingReviews } from "./useCopilotPendingReviews";

type Props =
  | { runs: RunReviewTarget[] }
  | {
      chatSessionId: string;
      pollWhileEmpty?: boolean;
      refetchKey?: number;
      expertName?: string | null;
      // Its avatar in the review dialog.
      expert?: ExpertIdentity | null;
    };

/**
 * Renders the chat's pending reviews, or those of the agent runs it started:
 * held AutoPilot calls in one "Waiting for you" queue, oldest first, and
 * every block, MCP or run review in one consolidated list.
 */
export function CopilotPendingReviews(props: Props) {
  const { onSend, onBackendTurn } = useCopilotChatActions();
  const runs = "runs" in props ? props.runs : null;
  const expertName = "expertName" in props ? props.expertName : null;
  const expert = ("expert" in props ? props.expert : null) ?? null;
  const { pendingReviews, answeredReviews, refetch } =
    useCopilotPendingReviews(props);

  const heldReviews = pendingReviews
    .filter(isGateReview)
    .sort((a, b) => +new Date(a.created_at) - +new Date(b.created_at));
  const otherReviews = pendingReviews.filter(
    (r) => !r.node_exec_id.startsWith(COPILOT_GATE_NODE_PREFIX),
  );

  async function handleHeldCallAnswered(turnFollows: boolean) {
    await refetch();
    if (turnFollows) onBackendTurn?.();
  }

  async function handleReviewComplete() {
    const shownRuns = runIDs(otherReviews);
    const decidedRuns = shownRuns.length
      ? shownRuns
      : (runs ?? []).map((run) => run.graphExecId);
    // Brief delay for the server to propagate the approval
    await new Promise((resolve) => setTimeout(resolve, 500));
    const remaining = (await refetch()).filter(
      (r) => !r.node_exec_id.startsWith(COPILOT_GATE_NODE_PREFIX),
    );

    if (remaining.length > 0) return;

    // Graph executions auto-resume after approval; chat reviews need resume_capability.
    if (runs) {
      const many = decidedRuns.length > 1;
      onSend(
        `All pending reviews have been processed. ` +
          `The workflow execution${many ? "s" : ""} will resume automatically for approved reviews. ` +
          `Use view_agent_output with ${decidedRuns.map((id) => `execution_id="${id}"`).join(", ")} to check the result${many ? "s" : ""}.`,
      );
    } else {
      onSend(
        `All pending reviews have been processed. ` +
          `For any approved reviews, call resume_capability with the corresponding review_id to execute them. ` +
          `For rejected reviews, no further action is needed.`,
      );
    }
  }

  return (
    <div className="flex flex-col gap-2 py-2 empty:hidden">
      <ApprovalQueue
        items={heldReviews.map(toApprovalItem)}
        reviews={heldReviews}
        answered={answeredReviews}
        expertName={expertName}
        expert={
          expert
            ? {
                id: expert.id,
                name: expert.name,
                role: expert.role ?? "",
                avatar_url: expert.avatarUrl,
              }
            : null
        }
        onAnswered={handleHeldCallAnswered}
        onContinued={() => onBackendTurn?.()}
      />
      {otherReviews.length > 0 && (
        <PendingReviewsList
          reviews={otherReviews}
          onReviewComplete={handleReviewComplete}
        />
      )}
    </div>
  );
}

function runIDs(reviews: PendingHumanReviewModel[]) {
  return [
    ...new Set(
      reviews.flatMap((r) => (r.graph_exec_id ? [r.graph_exec_id] : [])),
    ),
  ];
}
