"use client";

import { cn } from "@/lib/utils";
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area";
import { forwardRef } from "react";
import { ScrollBar } from "./components/ScrollBar";
import { ScrollToTopButton } from "./components/ScrollToTopButton";
import { useScrollArea } from "./useScrollArea";

interface Props
  extends React.ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.Root> {
  orientation?: "vertical" | "horizontal" | "both";
  /** Fades in a "Scroll to top" button once the viewport is scrolled 200px. */
  showScrollToTop?: boolean;
  viewportClassName?: string;
}

export const ScrollArea = forwardRef<
  React.ElementRef<typeof ScrollAreaPrimitive.Root>,
  Props
>(function ScrollArea(
  {
    className,
    viewportClassName,
    children,
    orientation = "vertical",
    showScrollToTop = false,
    ...props
  },
  ref,
) {
  const { viewportRef, isScrolledPastThreshold, scrollToTop } = useScrollArea({
    showScrollToTop,
  });

  return (
    <ScrollAreaPrimitive.Root
      ref={ref}
      className={cn("relative overflow-hidden", className)}
      {...props}
    >
      <ScrollAreaPrimitive.Viewport
        ref={viewportRef}
        className={cn("h-full w-full rounded-[inherit]", viewportClassName)}
      >
        {children}
      </ScrollAreaPrimitive.Viewport>
      {orientation !== "horizontal" ? (
        <ScrollBar orientation="vertical" />
      ) : null}
      {orientation !== "vertical" ? (
        <ScrollBar orientation="horizontal" />
      ) : null}
      <ScrollAreaPrimitive.Corner />
      {showScrollToTop ? (
        <ScrollToTopButton
          visible={isScrolledPastThreshold}
          onClick={scrollToTop}
        />
      ) : null}
    </ScrollAreaPrimitive.Root>
  );
});
