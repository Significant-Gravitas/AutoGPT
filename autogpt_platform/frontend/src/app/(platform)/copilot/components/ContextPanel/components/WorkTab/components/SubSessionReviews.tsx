"use client";

import { useState } from "react";
import { usePendingReviewsForChatSession } from "@/hooks/usePendingReviews";
import { useProcessReviews } from "@/hooks/useProcessReviews";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import {
  ApprovalCard,
  type CardStatus,
} from "../../../../ApprovalQueue/components/ApprovalCard/ApprovalCard";
import {
  type ChatRule,
  type RuleScope,
  isGateReview,
  toApprovalItem,
} from "../../../../ApprovalQueue/helpers";
import { DetailSection } from "./DetailSection";

interface CardProps {
  review: PendingHumanReviewModel;
  expertName: string;
}

function SubSessionReviewCard({ review, expertName }: CardProps) {
  const item = toApprovalItem(review);
  const { processReviews } = useProcessReviews();
  const [status, setStatus] = useState<CardStatus>("idle");
  const [failed, setFailed] = useState(false);

  async function answer(approved: boolean, rule?: ChatRule, scope?: RuleScope) {
    setStatus(approved ? "approving" : "rejecting");
    setFailed(false);
    try {
      const res = await processReviews(
        [
          {
            node_exec_id: item.reviewId,
            approved,
            chat_rule: approved ? (rule ?? null) : null,
            ...(approved && rule && scope ? { chat_rule_scope: scope } : {}),
          },
        ],
        [item.scope],
      );
      setFailed(!(res.status === 200 && res.data.failed_count === 0));
    } catch {
      setFailed(true);
    }
    setStatus("idle");
  }

  return (
    <div className="overflow-hidden rounded-xl border border-amber-200 bg-white">
      <ApprovalCard
        item={item}
        status={status}
        failed={failed}
        expertName={expertName}
        onApprove={(rule, scope) => void answer(true, rule, scope)}
        onReject={() => void answer(false)}
      />
    </div>
  );
}

interface Props {
  subSessionId: string | null;
  expertName: string;
}

/** The teammate's own held calls: without them here a teammate stopped at
 *  their gate looks stuck from Otto's chat. */
export function SubSessionReviews({ subSessionId, expertName }: Props) {
  const { pendingReviews } = usePendingReviewsForChatSession(
    subSessionId ?? "",
  );
  const reviews = pendingReviews.filter(isGateReview);
  if (!subSessionId || reviews.length === 0) return null;
  return (
    <DetailSection title={`${expertName} is waiting for approval`}>
      <div className="flex flex-col gap-2" data-testid="sub-session-reviews">
        {reviews.map((review) => (
          <SubSessionReviewCard
            key={review.node_exec_id}
            review={review}
            expertName={expertName}
          />
        ))}
      </div>
    </DetailSection>
  );
}
