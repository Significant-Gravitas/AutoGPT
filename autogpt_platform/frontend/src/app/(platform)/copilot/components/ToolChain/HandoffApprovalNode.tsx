"use client";

import { useContext, useState } from "react";
import { usePendingReviewsForChatSession } from "@/hooks/usePendingReviews";
import { useProcessReviews } from "@/hooks/useProcessReviews";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { cn } from "@/lib/utils";
import { ChatSessionContext } from "../ChatContainer/components/ChatSessionContext";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";
import {
  ApprovalCard,
  type CardStatus,
} from "../ApprovalQueue/components/ApprovalCard/ApprovalCard";
import {
  type ApprovalItem,
  type ChatRule,
  type RuleScope,
  isGateReview,
  toApprovalItem,
} from "../ApprovalQueue/helpers";
import { HeldCallDetail } from "./HeldCallRowParts";
import type { HeldRowInfo } from "./heldRow";

export const HANDOFF_TOOLS = new Set([
  "delegate_to_expert",
  "handoff_to_expert",
]);

/** A hand-off held for approval is a node on the wire, not a card under
 *  the chain: the wire runs into its top edge and out of its bottom. */
export function isHandoffApprovalRow(
  tool: string | undefined,
  held: HeldRowInfo | undefined,
): boolean {
  return !!tool && HANDOFF_TOOLS.has(tool) && held?.state === "waiting";
}

interface Props {
  held: HeldRowInfo;
  isLast: boolean;
  expertName?: string | null;
}

export function HandoffApprovalNode({
  held,
  isLast,
  expertName = null,
}: Props) {
  const sessionId = useContext(ChatSessionContext);
  const actions = useContext(CopilotChatActionsContext);
  const { pendingReviews, isLoading } = usePendingReviewsForChatSession(
    sessionId ?? "",
  );
  const { processReviews } = useProcessReviews();
  const [status, setStatus] = useState<CardStatus>("idle");
  const [failed, setFailed] = useState(false);
  const review = pendingReviews.find(
    (r) => isGateReview(r) && r.node_exec_id === held.reviewId,
  );
  const item: ApprovalItem | null = review ? toApprovalItem(review) : null;

  async function answer(approved: boolean, rule?: ChatRule, scope?: RuleScope) {
    if (!item) return;
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
          },
        ],
        [item.scope],
      );
      ok = res.status === 200 && res.data.failed_count === 0;
    } catch {
      ok = false;
    }
    setStatus("idle");
    if (ok) actions?.onBackendTurn?.();
    else setFailed(true);
  }

  return (
    <div className="flex flex-col" data-testid="handoff-approval-node">
      <span aria-hidden className="ml-[14px] h-3 w-px bg-zinc-200" />
      <div
        className={cn(
          "w-full overflow-hidden rounded-2xl border bg-white",
          item ? "border-amber-200" : "border-zinc-200",
        )}
      >
        {item ? (
          <ApprovalCard
            item={item}
            status={status}
            failed={failed}
            expertName={expertName}
            onApprove={(rule, scope) => void answer(true, rule, scope)}
            onReject={() => void answer(false)}
          />
        ) : isLoading ? (
          <div className="flex flex-col gap-2 px-4 py-3">
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-8 w-40 rounded-full" />
          </div>
        ) : (
          <div className="px-4 py-3">
            <HeldCallDetail held={held} />
          </div>
        )}
      </div>
      {!isLast && (
        <span aria-hidden className="ml-[14px] h-3 w-px bg-zinc-200" />
      )}
    </div>
  );
}
