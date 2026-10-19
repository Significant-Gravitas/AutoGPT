"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { cn } from "@/lib/utils";
import { DashboardSpeed02Icon } from "@hugeicons/core-free-icons";
import { formatTokenCount } from "../../tokenDevtool/tokenMath";
import { TokenDevtoolPanel } from "./components/TokenDevtoolPanel";
import { useTokenDevtoolBadge } from "./useTokenDevtoolBadge";

interface Props {
  sessionId: string;
  /** Alignment is the tray's concern — the badge renders into two
   *  structurally different parents. */
  className?: string;
}

/** Dev-only context readout in the composer tray. The bar tracks the
 *  estimated live context against the model window, with a marker at the
 *  backend's autocompact trigger — cross it and the next turn is expected
 *  to summarize. The popover splits the estimate by source and lists
 *  per-turn usage; ⟲ marks a turn where the stream carried a
 *  `context_compaction` tool call. */
export function TokenDevtoolBadge({ sessionId, className }: Props) {
  const devtool = useTokenDevtoolBadge(sessionId);
  const { context } = devtool;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={`Token devtool, context ${
            context === null ? "unknown" : formatTokenCount(context)
          }`}
          className={cn(
            "flex h-8 w-8 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700",
            className,
          )}
        >
          <Icon icon={DashboardSpeed02Icon} size={16} />
          <span className="sr-only">
            {context === null ? "ctx —" : `ctx ~${formatTokenCount(context)}`}
          </span>
        </button>
      </PopoverTrigger>

      <PopoverContent
        side="top"
        align="end"
        sideOffset={8}
        className="w-80 rounded-2xl border-zinc-200/70 p-3 shadow-[0_16px_40px_-24px_rgba(0,0,0,0.25)]"
      >
        <TokenDevtoolPanel {...devtool} />
      </PopoverContent>
    </Popover>
  );
}
