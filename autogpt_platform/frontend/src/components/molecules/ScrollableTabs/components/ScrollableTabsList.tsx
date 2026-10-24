"use client";

import { cn } from "@/lib/utils";
import * as React from "react";
import { useScrollableTabs } from "../context";

export const ScrollableTabsList = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(function ScrollableTabsList({ className, children, ...props }, ref) {
  const { activeValue } = useScrollableTabs();
  const [activeTabElement, setActiveTabElement] =
    React.useState<HTMLElement | null>(null);

  React.useEffect(() => {
    const activeButton = Array.from(
      document.querySelectorAll<HTMLElement>(
        '[data-scrollable-tab-trigger][data-value="' + activeValue + '"]',
      ),
    )[0];

    if (activeButton) {
      setActiveTabElement(activeButton);
    }
  }, [activeValue]);

  return (
    <div data-slot="scrollable-tabs-list-root" className="relative" ref={ref}>
      <div
        data-slot="scrollable-tabs-list"
        className={cn(
          "inline-flex w-full items-center justify-start border-b border-separator",
          className,
        )}
        {...props}
      >
        {children}
      </div>
      {activeTabElement && (
        <div
          data-slot="scrollable-tabs-indicator"
          className="transition-left transition-right absolute bottom-0 h-0.5 bg-brand-hover duration-200 ease-in-out"
          style={{
            left: activeTabElement.offsetLeft,
            width: activeTabElement.offsetWidth,
            willChange: "left, width",
          }}
        />
      )}
    </div>
  );
});

ScrollableTabsList.displayName = "ScrollableTabsList";
