import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import type { CardStatus } from "@/app/(platform)/copilot/components/ApprovalQueue/components/ApprovalCard/ApprovalCard";
import {
  type ChatRule,
  type RuleScope,
  toApprovalItem,
} from "@/app/(platform)/copilot/components/ApprovalQueue/helpers";
import type { HeldReceipt } from "../useHeldReview";
import { HeldCallDetail } from "./HeldCallDetail";
import { HeldCallRow } from "./HeldCallRow";
import { HeldReceiptRow } from "./HeldReceiptRow";

interface Props {
  item: HomeAttentionItem;
  receipt: HeldReceipt | null;
  status: CardStatus;
  failed: boolean;
  open: boolean;
  // Inside an Expert's group the header carries the avatar.
  avatarSize?: number | null;
  onToggle: () => void;
  onClose: () => void;
  onDecide: (approved: boolean, rule?: ChatRule, scope?: RuleScope) => void;
}

// One held AutoPilot call: a row, the chat's card in its place, or its receipt.
export function HeldCall({
  item,
  receipt,
  status,
  failed,
  open,
  avatarSize = 40,
  onToggle,
  onClose,
  onDecide,
}: Props) {
  const approval = toApprovalItem(item.review!);
  if (receipt) return <HeldReceiptRow approval={approval} receipt={receipt} />;
  if (open)
    return (
      <HeldCallDetail
        item={item}
        approval={approval}
        status={status}
        failed={failed}
        avatarSize={avatarSize}
        onClose={onClose}
        onDecide={onDecide}
      />
    );
  return (
    <HeldCallRow
      item={item}
      approval={approval}
      status={status}
      failed={failed}
      avatarSize={avatarSize}
      onOpen={onToggle}
      onDecide={onDecide}
    />
  );
}
