"use client";

import * as React from "react";
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ArrowUpIcon } from "@phosphor-icons/react";

import { cn } from "@/lib/utils";

interface ScrollAreaProps
  extends React.ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.Root> {
  orientation?: "vertical" | "horizontal" | "both";
  showScrollToTop?: boolean;
}

const ScrollArea = React.forwardRef<
  React.ElementRef<typeof ScrollAreaPrimitive.Root>,
  ScrollAreaProps
>(
  (
    {
      className,
      children,
      orientation = "vertical",
      showScrollToTop = false,
      ...props
    },
    ref,
  ) => {
    const viewportRef = React.useRef<HTMLDivElement | null>(null);
    const reduceMotion = useReducedMotion();
    const fabVisible = useScrolledPastThreshold(viewportRef, {
      enabled: showScrollToTop,
      threshold: 200,
    });

    function scrollToTop() {
      viewportRef.current?.scrollTo({
        top: 0,
        behavior: reduceMotion ? "auto" : "smooth",
      });
    }

    return (
      <ScrollAreaPrimitive.Root
        ref={ref}
        className={cn("relative", className)}
        {...props}
      >
        <ScrollAreaPrimitive.Viewport
          ref={viewportRef}
          className="h-full w-full rounded-[inherit]"
          style={{
            overflowX: orientation === "vertical" ? "hidden" : "scroll",
            overflowY: orientation === "horizontal" ? "hidden" : "scroll",
          }}
        >
          {children}
        </ScrollAreaPrimitive.Viewport>
        {orientation !== "horizontal" && <ScrollBar />}
        {orientation !== "vertical" && <ScrollBar orientation="horizontal" />}
        <ScrollAreaPrimitive.Corner />
        {showScrollToTop && (
          <ScrollToTopFab visible={fabVisible} onClick={scrollToTop} />
        )}
      </ScrollAreaPrimitive.Root>
    );
  },
);
ScrollArea.displayName = ScrollAreaPrimitive.Root.displayName;

const ScrollBar = React.forwardRef<
  React.ElementRef<typeof ScrollAreaPrimitive.ScrollAreaScrollbar>,
  React.ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.ScrollAreaScrollbar>
>(({ className, orientation = "vertical", ...props }, ref) => (
  <ScrollAreaPrimitive.ScrollAreaScrollbar
    ref={ref}
    orientation={orientation}
    className={cn(
      "flex touch-none select-none transition-colors",
      orientation === "vertical" &&
        "h-full w-2.5 border-l border-l-transparent p-[1px]",
      orientation === "horizontal" &&
        "h-2.5 flex-col border-t border-t-transparent p-[1px]",
      className,
    )}
    {...props}
  >
    <ScrollAreaPrimitive.ScrollAreaThumb className="relative flex-1 rounded-full bg-neutral-200 dark:bg-neutral-800" />
  </ScrollAreaPrimitive.ScrollAreaScrollbar>
));
ScrollBar.displayName = ScrollAreaPrimitive.ScrollAreaScrollbar.displayName;

function useScrolledPastThreshold(
  viewportRef: React.RefObject<HTMLDivElement | null>,
  { enabled, threshold }: { enabled: boolean; threshold: number },
) {
  const [visible, setVisible] = React.useState(false);

  React.useEffect(() => {
    if (!enabled) return;
    const viewport = viewportRef.current;
    if (!viewport) return;

    function update() {
      // TS can't narrow the captured `viewport` inside a nested closure, so
      // keep the guard — the outer early-return still covers runtime.
      if (!viewport) return;
      setVisible(viewport.scrollTop > threshold);
    }

    update();
    viewport.addEventListener("scroll", update, { passive: true });
    return () => viewport.removeEventListener("scroll", update);
  }, [enabled, threshold, viewportRef]);

  return visible;
}

interface ScrollToTopFabProps {
  visible: boolean;
  onClick: () => void;
}

function ScrollToTopFab({ visible, onClick }: ScrollToTopFabProps) {
  const reduceMotion = useReducedMotion();

  return (
    <AnimatePresence>
      {visible && (
        <motion.button
          type="button"
          onClick={onClick}
          aria-label="Scroll to top"
          className="absolute bottom-6 left-1/2 z-30 flex h-11 w-11 -translate-x-1/2 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-md transition-colors hover:bg-primary/90 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          initial={
            reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.95, y: 8 }
          }
          animate={
            reduceMotion ? { opacity: 1 } : { opacity: 1, scale: 1, y: 0 }
          }
          exit={
            reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.95, y: 8 }
          }
          transition={{ duration: 0.15, ease: [0, 0, 0.2, 1] }}
        >
          <ArrowUpIcon size={20} weight="bold" />
        </motion.button>
      )}
    </AnimatePresence>
  );
}

export { ScrollArea, ScrollBar };
