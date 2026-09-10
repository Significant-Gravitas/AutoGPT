import { Cancel01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

interface Props {
  item: HomeAttentionItem;
  isProcessing: boolean;
  confirmDecline: boolean;
  onApprove: () => void;
  onDecline: () => void;
  onDeclineBlur: () => void;
}

const ICON_BUTTON_CLASS = "size-7 rounded-md p-0";

export function AttentionRowActions({
  item,
  isProcessing,
  confirmDecline,
  onApprove,
  onDecline,
  onDeclineBlur,
}: Props) {
  const primaryAction = (
    <Button
      as="NextLink"
      href={item.primary_action.href}
      variant="secondary"
      size="small"
      className="h-7 min-w-0 rounded-md px-2.5 text-xs"
    >
      {item.primary_action.label}
    </Button>
  );

  if (item.kind !== "approval" || !item.review) {
    return primaryAction;
  }

  return (
    <>
      {primaryAction}
      <Button
        variant="icon"
        size="icon"
        className={cn(
          ICON_BUTTON_CLASS,
          "border-zinc-900 bg-zinc-900 text-white hover:border-zinc-800 hover:bg-zinc-800",
        )}
        disabled={isProcessing}
        aria-label={`Approve: ${item.title}`}
        onClick={onApprove}
      >
        <Icon icon={Tick02Icon} size={15} aria-hidden="true" />
      </Button>
      <Button
        variant="icon"
        size="icon"
        className={cn(
          ICON_BUTTON_CLASS,
          "border-zinc-200 text-zinc-600 hover:border-zinc-300 hover:bg-zinc-50",
          confirmDecline &&
            "border-red-500 bg-red-500 text-white hover:border-red-600 hover:bg-red-600 hover:text-white",
        )}
        disabled={isProcessing}
        aria-label={`${confirmDecline ? "Confirm decline" : "Decline"}: ${item.title}`}
        onClick={onDecline}
        onBlur={onDeclineBlur}
      >
        <Icon icon={Cancel01Icon} size={15} aria-hidden="true" />
      </Button>
    </>
  );
}
