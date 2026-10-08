"use client";

import { ScrollBar } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { ScrollArea as ScrollAreaPrimitive } from "@base-ui/react/scroll-area";
import type { ComponentProps } from "react";
import { ScrollToTopButton } from "./components/ScrollToTopButton";
import { useScrollArea } from "./useScrollArea";

interface Props extends Omit<
  ComponentProps<typeof ScrollAreaPrimitive.Root>,
  "className"
> {
  className?: string;
  orientation?: "vertical" | "horizontal" | "both";
  /** Fades in a "Scroll to top" button once the viewport is scrolled 200px. */
  showScrollToTop?: boolean;
  viewportClassName?: string;
}

// Composed from the Base UI parts rather than Kobra's `ScrollArea` wrapper,
// which owns its viewport: the atom needs a ref on it for scroll-to-top, a
// class hook for callers, and a horizontal scrollbar.
export function ScrollArea({
  className,
  viewportClassName,
  children,
  orientation = "vertical",
  showScrollToTop = false,
  ...props
}: Props) {
  const { viewportRef, isScrolledPastThreshold, scrollToTop } = useScrollArea({
    showScrollToTop,
  });

  return (
    <ScrollAreaPrimitive.Root
      data-slot="scroll-area"
      className={cn("relative overflow-hidden", className)}
      {...props}
    >
      <ScrollAreaPrimitive.Viewport
        ref={viewportRef}
        data-slot="scroll-area-viewport"
        className={cn(
          "size-full rounded-[inherit] outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-1",
          viewportClassName,
        )}
      >
        {children}
      </ScrollAreaPrimitive.Viewport>
      {orientation !== "horizontal" ? (
        <ScrollBar orientation="vertical" />
      ) : null}
      {orientation !== "vertical" ? (
        <ScrollBar orientation="horizontal" />
      ) : null}
      <ScrollAreaPrimitive.Corner data-slot="scroll-area-corner" />
      {showScrollToTop ? (
        <ScrollToTopButton
          visible={isScrolledPastThreshold}
          onClick={scrollToTop}
        />
      ) : null}
    </ScrollAreaPrimitive.Root>
  );
}
