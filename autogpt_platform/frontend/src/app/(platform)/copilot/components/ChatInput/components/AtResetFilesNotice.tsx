"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { Alert02Icon } from "@hugeicons/core-free-icons";

interface Props {
  className?: string;
}

/** Why Send at reset is off while files are attached: a message set for the
 *  reset carries text only. */
export function AtResetFilesNotice({ className }: Props) {
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
        Only text can wait for the reset. Remove the files to schedule this
        message.
      </span>
    </div>
  );
}
