"use client";

import { cn } from "@/lib/utils";
import { MAX_SEARCH_RESULTS } from "./helpers";

// Mirrors KitResultRow's geometry so results drop straight into the
// skeleton's footprint — no reflow when the query resolves.
export function KitSearchSkeleton() {
  return (
    <div
      role="status"
      aria-label="Searching"
      className="w-full max-w-[42rem] overflow-hidden rounded-2xl border border-border bg-background duration-200 animate-in fade-in motion-reduce:animate-none"
    >
      {Array.from({ length: MAX_SEARCH_RESULTS }, (_, index) => (
        <div
          key={index}
          className="flex items-center gap-3 border-b border-border px-3.5 py-3 last:border-b-0"
        >
          <ShimmerBar delay={index} className="size-9 shrink-0 rounded-xl" />
          <div className="flex min-w-0 flex-1 flex-col gap-1.5">
            <ShimmerBar delay={index} className="h-3 w-1/3 rounded-full" />
            <ShimmerBar delay={index} className="h-2.5 w-1/5 rounded-full" />
          </div>
          <ShimmerBar delay={index} className="h-7 w-16 shrink-0 rounded-xl" />
        </div>
      ))}
    </div>
  );
}

interface ShimmerBarProps {
  delay: number;
  className?: string;
}

// The shared `animate-shimmer` keyframes run on a slow ambient clock; a
// skeleton needs a tighter linear sweep, and the per-row delay keeps the
// three rows out of phase so the block reads as alive rather than blinking.
function ShimmerBar({ delay, className }: ShimmerBarProps) {
  return (
    <span
      aria-hidden
      style={{ animationDelay: `${delay * 140}ms` }}
      className={cn(
        "block animate-shimmer bg-[linear-gradient(90deg,theme(colors.zinc.100)_25%,theme(colors.zinc.50)_50%,theme(colors.zinc.100)_75%)] bg-[length:200%_100%] [animation-duration:1.4s] [animation-timing-function:linear] motion-reduce:animate-none motion-reduce:bg-zinc-100",
        className,
      )}
    />
  );
}
