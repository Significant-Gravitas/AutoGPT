"use client";

import { useContext, useState } from "react";
import { usePendingReviewsForChatSession } from "@/hooks/usePendingReviews";
import { useProcessReviews } from "@/hooks/useProcessReviews";
import { useHeldAnswersStore } from "../ApprovalQueue/heldAnswersStore";
import { ChatSessionContext } from "../ChatContainer/components/ChatSessionContext";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";
import type { CardStatus } from "../ApprovalQueue/components/ApprovalCard/ApprovalCard";
import {
  type ChatRule,
  type RuleScope,
  isGateReview,
  toApprovalItem,
} from "../ApprovalQueue/helpers";
import {
  editedReview,
  type HandoffEdits,
} from "../HandoffApprovalCard/helpers";
import type { HeldRowInfo } from "./heldRow";

/** The held hand-off's review, and answering it from the wire. */
export function useHandoffApproval(held: HeldRowInfo) {
  const sessionId = useContext(ChatSessionContext);
  const actions = useContext(CopilotChatActionsContext);
  const { pendingReviews, isLoading } = usePendingReviewsForChatSession(
    sessionId ?? "",
  );
  const { processReviews } = useProcessReviews();
  const recordAnswers = useHeldAnswersStore((state) => state.record);
  const [status, setStatus] = useState<CardStatus>("idle");
  const [failed, setFailed] = useState(false);
  const review = pendingReviews.find(
    (r) => isGateReview(r) && r.node_exec_id === held.reviewId,
  );
  const item = review ? toApprovalItem(review) : null;

  async function answer(
    approved: boolean,
    edits: HandoffEdits = {},
    rule?: ChatRule,
    scope?: RuleScope,
  ) {
    if (!item || !review) return;
    setStatus(approved ? "approving" : "rejecting");
    setFailed(false);
    let ok = false;
    try {
      const res = await processReviews(
        [
          {
            node_exec_id: item.reviewId,
            approved,
            chat_rule: approved ? (rule ?? null) : null,
            ...(approved && rule && scope ? { chat_rule_scope: scope } : {}),
            ...(approved && review.editable ? editedReview(item, edits) : {}),
          },
        ],
        [item.scope],
      );
      ok = res.status === 200 && res.data.failed_count === 0;
    } catch {
      ok = false;
    }
    setStatus("idle");
    if (ok) {
      // The row flips to the answer now; the persisted result lands later.
      recordAnswers([item.reviewId], approved);
      actions?.onBackendTurn?.();
    } else setFailed(true);
  }

  return { review, item, isLoading, status, failed, answer };
}
