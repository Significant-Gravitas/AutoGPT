"use client";

import { cn } from "@/lib/utils";
import type { ChainRow } from "../../../../ToolChain/helpers";
import { RowIcon } from "../../../../ToolChain/RowIcon";
import { SwapText } from "../../../../ToolChain/SwapText";

interface Props {
  steps: ChainRow[];
  latestText: string | null;
}

const MAX_STEPS = 6;

/** The teammate's recent steps, drawn like the chat's own chain in
 *  miniature, with their latest words underneath. */
export function DelegationTimeline({ steps, latestText }: Props) {
  const recent = steps.slice(-MAX_STEPS);
  if (recent.length === 0 && !latestText) {
    return <p className="text-sm text-zinc-500">Nothing to show yet.</p>;
  }
  return (
    <div className="flex flex-col">
      {recent.map((row, i) => {
        const isLast = i === recent.length - 1;
        return (
          <div key={row.key} className="flex items-stretch gap-2.5">
            <div className="flex w-6 flex-col items-center">
              <div className="flex size-6 shrink-0 items-center justify-center rounded-full bg-zinc-100">
                <RowIcon row={row} />
              </div>
              {!isLast && <div className="w-px flex-1 bg-zinc-200" />}
            </div>
            <div className={cn("min-w-0 flex-1 pt-[2px]", !isLast && "pb-2.5")}>
              <SwapText
                text={row.text}
                shimmer={row.state === "running"}
                className="max-w-full text-sm leading-5 text-zinc-600"
              />
            </div>
          </div>
        );
      })}
      {latestText && (
        <p className="mt-2 line-clamp-4 text-sm leading-relaxed text-zinc-600">
          {latestText}
        </p>
      )}
    </div>
  );
}
