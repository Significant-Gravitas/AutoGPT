import {
  Cancel01Icon,
  InformationCircleIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { Receipt } from "../useApprovalQueue";
import { HeadlineText } from "./ApprovalHeadline";

interface Props {
  receipt: Receipt;
}

export function ReceiptLine({ receipt }: Props) {
  const [icon, tone] =
    receipt.text === "Approved" || receipt.text === "Released"
      ? [Tick02Icon, "text-success"]
      : receipt.text === "Rejected" || receipt.text === "Kept out"
        ? [Cancel01Icon, "text-foreground-subtle"]
        : [InformationCircleIcon, "text-foreground-subtle"];
  return (
    <li className="flex items-center gap-2 px-4 py-2 text-sm text-foreground-secondary">
      <Icon icon={icon} size={14} className={tone} aria-hidden />
      <span className="min-w-0 truncate">
        <HeadlineText item={receipt.item} />
      </span>
      <span className="shrink-0 text-foreground-subtle">· {receipt.text}</span>
    </li>
  );
}
