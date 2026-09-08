import { Cancel01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { Button } from "@/components/atoms/Button/Button";

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
      size="xs"
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
        variant="primary"
        size="icon-xs"
        leadingIcon={Tick02Icon}
        disabled={isProcessing}
        aria-label={`Approve: ${item.title}`}
        onClick={onApprove}
      />
      <Button
        variant={confirmDecline ? "destructive" : "icon"}
        size="icon-xs"
        leadingIcon={Cancel01Icon}
        disabled={isProcessing}
        aria-label={`${confirmDecline ? "Confirm decline" : "Decline"}: ${item.title}`}
        onClick={onDecline}
        onBlur={onDeclineBlur}
      />
    </>
  );
}
