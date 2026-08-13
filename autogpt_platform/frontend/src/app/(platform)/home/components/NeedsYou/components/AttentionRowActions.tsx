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
        className="size-10 border-zinc-800 bg-zinc-800 p-0 text-white hover:border-zinc-900 hover:bg-zinc-900"
        disabled={isProcessing}
        aria-label={`Approve: ${item.title}`}
        onClick={onApprove}
      >
        <Icon icon={Tick02Icon} size={18} aria-hidden="true" />
      </Button>
      <Button
        variant="icon"
        size="icon"
        className={cn(
          "size-10 p-0",
          confirmDecline &&
            "border-red-500 bg-red-500 text-white hover:border-red-600 hover:bg-red-600",
        )}
        disabled={isProcessing}
        aria-label={`${confirmDecline ? "Confirm decline" : "Decline"}: ${item.title}`}
        onClick={onDecline}
        onBlur={onDeclineBlur}
      >
        <Icon icon={Cancel01Icon} size={18} aria-hidden="true" />
      </Button>
    </>
  );
}
