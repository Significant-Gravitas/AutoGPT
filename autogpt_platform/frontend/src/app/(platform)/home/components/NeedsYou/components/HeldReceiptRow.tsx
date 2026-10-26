import {
  Cancel01Icon,
  InformationCircleIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { HeadlineText } from "@/app/(platform)/copilot/components/ApprovalQueue/components/ApprovalHeadline";
import type { ApprovalItem } from "@/app/(platform)/copilot/components/ApprovalQueue/helpers";
import type { HeldReceipt } from "../useHeldReview";

interface Props {
  approval: ApprovalItem;
  receipt: HeldReceipt;
  // In the review dialog the receipt stands where the card's heading was.
  heading?: boolean;
}

const ICONS = {
  approved: [Tick02Icon, "text-success"],
  rejected: [Cancel01Icon, "text-foreground-subtle"],
  elsewhere: [InformationCircleIcon, "text-foreground-subtle"],
} as const;

export function HeldReceiptRow({ approval, receipt, heading = false }: Props) {
  const [icon, tone] = ICONS[receipt.outcome];
  return (
    <div className="flex items-center gap-2 px-4 py-2 text-sm text-foreground-secondary">
      <Icon icon={icon} size={14} className={tone} aria-hidden />
      {heading ? (
        <h3 className="min-w-0 truncate">
          <HeadlineText item={approval} />
        </h3>
      ) : (
        <span className="min-w-0 truncate">
          <HeadlineText item={approval} />
        </span>
      )}
      <span className="shrink-0 text-foreground-subtle">· {receipt.text}</span>
    </div>
  );
}
