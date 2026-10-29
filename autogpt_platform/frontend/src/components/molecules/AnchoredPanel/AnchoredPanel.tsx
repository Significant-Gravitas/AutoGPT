"use client";

import * as PopperPrimitive from "@radix-ui/react-popper";
import * as React from "react";
import { createPortal } from "react-dom";

import { cn } from "@/lib/utils";

type Measurable = { getBoundingClientRect(): DOMRect };

interface Props
  extends React.ComponentPropsWithoutRef<typeof PopperPrimitive.Content> {
  /** What the panel sits beside: an element's box, or any measured box. */
  virtualRef: React.RefObject<Measurable | null>;
}

// A panel positioned beside something, like PopoverContent, but with no
// dismiss layer: it takes no focus, closes on nothing by itself, and leaves
// Escape and outside clicks to whatever they were meant for. For hints that
// sit next to the page while the person keeps working.
export const AnchoredPanel = React.forwardRef<HTMLDivElement, Props>(
  (
    { virtualRef, className, align = "center", sideOffset = 4, ...props },
    ref,
  ) => {
    if (typeof document === "undefined") return null;
    return (
      <PopperPrimitive.Root>
        <PopperPrimitive.Anchor virtualRef={virtualRef} />
        {createPortal(
          <PopperPrimitive.Content
            ref={ref}
            align={align}
            sideOffset={sideOffset}
            data-slot="anchored-panel"
            className={cn(
              "z-50 w-72 rounded-lg border border-border-base bg-surface p-4 text-foreground-strong shadow-md outline-none animate-in fade-in-0 zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
              className,
            )}
            {...props}
          />,
          document.body,
        )}
      </PopperPrimitive.Root>
    );
  },
);
AnchoredPanel.displayName = "AnchoredPanel";
