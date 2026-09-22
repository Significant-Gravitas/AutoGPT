"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { MAX_ATTACHMENTS } from "../../../helpers/workspaceAttachments";
import { cn } from "@/lib/utils";
import { Alert02Icon, Cancel01Icon } from "@hugeicons/core-free-icons";

interface Props {
  /** How many attachments the cap turned away on the last attach. */
  refusedCount: number;
  onDismiss: () => void;
  className?: string;
}

/**
 * Says what the cap refused, in the composer, so the draft the user is typing
 * is never interrupted and never outlived by a toast.
 */
export function AttachmentCapNotice({
  refusedCount,
  onDismiss,
  className,
}: Props) {
  return (
    <div
      role="status"
      className={cn(
        "flex w-full items-center gap-1.5 rounded-2xl bg-amber-50 px-3 py-1.5 text-sm text-amber-800",
        className,
      )}
    >
      <Icon icon={Alert02Icon} className="h-4 w-4 shrink-0" aria-hidden />
      <span className="min-w-0 flex-1">
        Up to {MAX_ATTACHMENTS} attachments per message — {refusedCount} not
        added
      </span>
      <Button
        type="button"
        variant="icon"
        size="icon-xs"
        aria-label="Dismiss attachment limit notice"
        onClick={onDismiss}
      >
        <Icon icon={Cancel01Icon} className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}
