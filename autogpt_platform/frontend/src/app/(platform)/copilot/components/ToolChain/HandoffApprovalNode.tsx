"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { HandoffApprovalCard } from "../HandoffApprovalCard/HandoffApprovalCard";
import { HeldCallDetail } from "./HeldCallRowParts";
import type { HeldRowInfo } from "./heldRow";
import { useHandoffApproval } from "./useHandoffApproval";
import { WireNode } from "./WireNode";

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
}

export function HandoffApprovalNode({ held, isLast }: Props) {
  const { review, item, isLoading, status, failed, answer } =
    useHandoffApproval(held);

  return (
    <WireNode isLast={isLast} testId="handoff-approval-node">
      {item && review ? (
        <HandoffApprovalCard
          item={item}
          payload={review.payload}
          editable={review.editable}
          status={status}
          failed={failed}
          onApprove={(edits, rule, scope) =>
            void answer(true, edits, rule, scope)
          }
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
    </WireNode>
  );
}
