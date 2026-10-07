import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import type { CardStatus } from "@/app/(platform)/copilot/components/ApprovalQueue/components/ApprovalCard/ApprovalCard";
import { toApprovalItem } from "@/app/(platform)/copilot/components/ApprovalQueue/helpers";
import type { HeldReceipt } from "../useHeldReview";
import { HeldCallRow } from "./HeldCallRow";
import { HeldReceiptRow } from "./HeldReceiptRow";

interface Props {
  item: HomeAttentionItem;
  receipt: HeldReceipt | null;
  status: CardStatus;
  failed: boolean;
  // Inside an Expert's group the header carries the avatar.
  avatarSize?: number | null;
  onOpen: () => void;
  onDecide: (approved: boolean) => void;
}

// One held AutoPilot call on the list: its row, or its receipt once decided.
export function HeldCall({
  item,
  receipt,
  status,
  failed,
  avatarSize = 40,
  onOpen,
  onDecide,
}: Props) {
  const approval = toApprovalItem(item.review!);
  if (receipt) return <HeldReceiptRow approval={approval} receipt={receipt} />;
  return (
    <HeldCallRow
      item={item}
      approval={approval}
      status={status}
      failed={failed}
      avatarSize={avatarSize}
      onOpen={onOpen}
      onDecide={onDecide}
    />
  );
}
