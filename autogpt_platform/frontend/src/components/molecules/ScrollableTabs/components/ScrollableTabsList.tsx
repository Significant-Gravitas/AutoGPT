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
    <div className="relative" ref={ref}>
      <div
        className={cn(
          "inline-flex w-full items-center justify-start border-b border-zinc-100",
          className,
        )}
        {...props}
      >
        {children}
      </div>
      {activeTabElement && (
        <div
          className="transition-left transition-right absolute bottom-0 h-0.5 bg-purple-600 duration-200 ease-in-out"
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
