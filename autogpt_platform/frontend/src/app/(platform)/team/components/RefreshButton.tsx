"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { RefreshIcon } from "@hugeicons/core-free-icons";

interface Props {
  onRefresh: () => void;
  isRefreshing: boolean;
}

/** Sized and weighted like the filter strip's "more" trigger it sits next to,
 *  so the two read as one row of quiet icon controls. */
export function RefreshButton({ onRefresh, isRefreshing }: Props) {
  return (
    <button
      type="button"
      aria-label="Refresh"
      disabled={isRefreshing}
      onClick={onRefresh}
      className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-100 disabled:opacity-60"
    >
      <Icon
        icon={RefreshIcon}
        size={14}
        className={cn(isRefreshing && "animate-spin")}
      />
    </button>
  );
}
