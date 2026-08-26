"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { cn } from "@/lib/utils";
import { DashboardSpeed02Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { useTokenDevtoolStore } from "../../tokenDevtool/store";
import {
  AUTOCOMPACT_TOKENS,
  MODEL_CONTEXT_WINDOW,
  breakdownTotal,
  displayContext,
  formatTokenCount,
} from "../../tokenDevtool/tokenMath";
import { BreakdownSection } from "./components/BreakdownSection";
import { ContextBar } from "./components/ContextBar";
import { MiniBar } from "./components/MiniBar";
import { TurnRow } from "./components/TurnRow";

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
  const turns = useTokenDevtoolStore((s) => s.turnsBySession[sessionId]);
  const breakdown = useTokenDevtoolStore(
    (s) => s.breakdownBySession[sessionId],
  );
  const liveContext = useTokenDevtoolStore(
    (s) => s.liveContextBySession[sessionId],
  );
  const compacted = useTokenDevtoolStore(
    (s) => s.compactedBySession[sessionId] ?? false,
  );
  const [open, setOpen] = useState(false);
  const seed = breakdown ? breakdownTotal(breakdown) : undefined;
  const context = displayContext(liveContext ?? null, compacted, seed);
  // The rows describe the loaded-history seed, so showing them next to a
  // headline sourced from the live cache-write sum would read as one total
  // that does not add up. Only render them while the seed IS the headline.
  const showBreakdown = breakdown !== undefined && context === seed;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={`Token devtool, context ${
            context === null ? "unknown" : formatTokenCount(context)
          }`}
          className={cn(
            "flex items-center gap-1.5 rounded-full px-2 py-0.5 font-mono text-xs text-zinc-500 transition-colors hover:bg-zinc-200/60 hover:text-zinc-700",
            className,
          )}
        >
          <Icon icon={DashboardSpeed02Icon} size={14} />
          {context === null ? "ctx —" : `ctx ~${formatTokenCount(context)}`}
          {context !== null && <MiniBar context={context} />}
        </button>
      </PopoverTrigger>

      <PopoverContent
        side="top"
        align="end"
        sideOffset={8}
        className="w-80 rounded-2xl border-zinc-200/70 p-3 shadow-[0_16px_40px_-24px_rgba(0,0,0,0.25)]"
      >
        <div className="flex items-baseline justify-between pb-2">
          <span className="text-sm font-medium text-zinc-800">
            Context window
          </span>
          <span className="font-mono text-xs text-zinc-500">
            {context === null ? "—" : `~${formatTokenCount(context)}`} /{" "}
            {formatTokenCount(MODEL_CONTEXT_WINDOW)}
          </span>
        </div>
        <ContextBar context={context ?? 0} />
        <p className="pt-1 font-mono text-[10px] text-zinc-400">
          assumes a {formatTokenCount(MODEL_CONTEXT_WINDOW)} window; the backend
          threshold is configurable
        </p>
        <p className="pb-2.5 pt-0.5 text-right font-mono text-xs text-amber-500">
          summarizes ~{formatTokenCount(AUTOCOMPACT_TOKENS)}
        </p>

        {showBreakdown && <BreakdownSection breakdown={breakdown} />}

        {!turns?.length ? (
          <p className="text-sm text-zinc-500">
            Live per-turn data starts with your next message.
          </p>
        ) : (
          <div className="flex max-h-40 flex-col gap-1 overflow-y-auto">
            {turns.map((turn, i) => (
              <TurnRow key={`${turn.at}-${i}`} index={i} turn={turn} />
            ))}
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}
