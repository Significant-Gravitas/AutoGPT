"use client";

import { cn } from "@/lib/utils";

import { useSwap } from "./useSwap";

interface Props {
  /** What a swap is measured by. Defaults to the text being shown. */
  swapKey?: string;
  children: React.ReactNode;
  className?: string;
}

/**
 * Content that slides and blurs out when it changes, and its replacement in.
 *
 * Switching connection rewrites several things at once — the chip's label and
 * its glyph, and the model under each tier. Swapped in place they read as a
 * glitch; moving them makes the change legible as one thing replacing another.
 */
export function Swap({ swapKey, children, className }: Props) {
  const { shown, phase } = useSwap(swapKey ?? String(children), children);

  return (
    <span
      className={cn(
        "inline-block transition-[transform,filter,opacity] duration-150 ease-in-out motion-reduce:transition-none",
        // oxlint-disable-next-line design-system/no-arbitrary-values -- a 2px motion blur on the swap; blur-sm is 4px
        phase === "exit" && "-translate-y-1 opacity-0 blur-[2px]",
        phase === "enter" &&
          // oxlint-disable-next-line design-system/no-arbitrary-values -- a 2px motion blur on the swap; blur-sm is 4px
          "translate-y-1 opacity-0 blur-[2px] transition-none",
        className,
      )}
    >
      {shown}
    </span>
  );
}
