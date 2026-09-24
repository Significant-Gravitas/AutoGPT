import { Cancel01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { Receipt } from "../useApprovalQueue";
import { HeadlineText } from "./ApprovalHeadline";

interface Props {
  receipt: Receipt;
}

export function ReceiptLine({ receipt }: Props) {
  const positive = ["Approved", "Released"].includes(receipt.text);
  return (
    <li className="flex items-center gap-2 px-4 py-2 text-sm text-zinc-600">
      <Icon
        icon={positive ? Tick02Icon : Cancel01Icon}
        size={14}
        className={positive ? "text-green-600" : "text-zinc-400"}
        aria-hidden
      />
      <span className="min-w-0 truncate">
        <HeadlineText item={receipt.item} />
      </span>
      <span className="shrink-0 text-zinc-400">· {receipt.text}</span>
    </li>
  );
}
