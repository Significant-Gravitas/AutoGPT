import {
  Cancel01Icon,
  InformationCircleIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
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
  approved: [Tick02Icon, "text-green-600"],
  rejected: [Cancel01Icon, "text-zinc-400"],
  elsewhere: [InformationCircleIcon, "text-zinc-400"],
} as const;

export function HeldReceiptRow({ approval, receipt, heading = false }: Props) {
  const [icon, tone] = ICONS[receipt.outcome];
  return (
    <div className="flex items-center gap-2 px-4 py-2 text-sm text-zinc-600">
      <Icon icon={icon} size={14} className={tone} aria-hidden />
      {heading ? (
        <Text
          variant="body"
          as="h3"
          tone="secondary"
          unmask={false}
          className="min-w-0 truncate leading-5"
        >
          <HeadlineText item={approval} />
        </Text>
      ) : (
        <span className="min-w-0 truncate">
          <HeadlineText item={approval} />
        </span>
      )}
      <span className="shrink-0 text-zinc-400">· {receipt.text}</span>
    </div>
  );
}
