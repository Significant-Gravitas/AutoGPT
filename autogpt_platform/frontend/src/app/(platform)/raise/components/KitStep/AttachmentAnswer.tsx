"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { Forward02Icon } from "@hugeicons/core-free-icons";
import type { RaiseAttachmentDraft } from "../../helpers";
import { bubbleClassFor } from "../ColorStep/helpers";

interface Props {
  attachments: RaiseAttachmentDraft[];
  color: string | null;
}

export function AttachmentAnswer({ attachments, color }: Props) {
  const skipped = attachments.length === 0;
  return (
    <div
      className={cn(
        "ml-auto max-w-[80%] rounded-2xl border px-4 py-3 text-[15px] leading-relaxed text-foreground",
        skipped && "flex w-fit items-center gap-2",
        bubbleClassFor(color) ?? "border-accent bg-accent/5",
      )}
    >
      {skipped ? (
        <>
          <Icon icon={Forward02Icon} size={16} aria-hidden />
          Skipped
        </>
      ) : (
        attachments.map((attachment) => attachment.name).join(", ")
      )}
    </div>
  );
}
