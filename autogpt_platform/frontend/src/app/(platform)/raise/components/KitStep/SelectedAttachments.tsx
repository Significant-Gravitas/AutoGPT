"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import type { RaiseAttachmentDraft } from "../../helpers";
import { bubbleClassFor } from "../ColorStep/helpers";
import { attachmentKey } from "./useAttachmentPicker";

interface Props {
  attachments: RaiseAttachmentDraft[];
  color: string | null;
  onRemove: (key: string) => void;
}

export function SelectedAttachments({ attachments, color, onRemove }: Props) {
  if (attachments.length === 0) return null;
  return (
    <div
      role="list"
      aria-label="Selected tools"
      className="flex w-full max-w-[42rem] flex-wrap justify-end gap-2"
    >
      {attachments.map((attachment) => (
        <button
          key={attachmentKey(attachment)}
          type="button"
          onClick={() => onRemove(attachmentKey(attachment))}
          className={cn(
            "group inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-sm text-foreground",
            "transition-transform duration-200 hover:scale-[1.03] active:scale-95 motion-reduce:transition-none motion-reduce:hover:scale-100",
            // Chips pop in as they are picked, so the jump from the results
            // list up to the tray reads as one continuous move.
            "duration-300 animate-in fade-in zoom-in-95 fill-mode-both motion-reduce:animate-none",
            bubbleClassFor(color) ?? "border-accent bg-accent/5",
          )}
        >
          {attachment.name}
          <Icon
            icon={Cancel01Icon}
            size={12}
            aria-hidden
            className="opacity-50 transition-opacity duration-200 group-hover:opacity-100 motion-reduce:transition-none"
          />
          <span className="sr-only">Remove {attachment.name}</span>
        </button>
      ))}
    </div>
  );
}
