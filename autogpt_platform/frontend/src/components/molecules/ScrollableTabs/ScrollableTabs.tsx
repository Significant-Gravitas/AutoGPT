"use client";

import { cn } from "@/lib/utils";
import { Children, useEffect, useRef, useState } from "react";
import { ScrollableTabsContent } from "./components/ScrollableTabsContent";
import { ScrollableTabsList } from "./components/ScrollableTabsList";
import { ScrollableTabsTrigger } from "./components/ScrollableTabsTrigger";
import { ScrollableTabsContext } from "./context";
import { findContentElements, findListElement } from "./helpers";
import { useScrollableTabsInternal } from "./useScrollableTabs";

interface Props {
  children?: React.ReactNode;
  className?: string;
  defaultValue?: string;
}

export function ScrollableTabs({ children, className, defaultValue }: Props) {
  const {
    activeValue,
    setActiveValue,
    registerContent,
    scrollToSection,
    scrollContainer,
    contentContainerRef,
  } = useScrollableTabsInternal({ defaultValue });

  const [shouldShowTabs, setShouldShowTabs] = useState(false);
  const measureRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!measureRef.current) return;

    function checkHeight() {
      if (measureRef.current) {
        const height = measureRef.current.scrollHeight;
        setShouldShowTabs(height > 800);
      }
    }

    checkHeight();

    const resizeObserver = new ResizeObserver(checkHeight);
    resizeObserver.observe(measureRef.current);

    return () => {
      resizeObserver.disconnect();
    };
  }, [children]);

  const childrenArray = Children.toArray(children);
  const listElement = findListElement(childrenArray);
  const contentElements = findContentElements(childrenArray);

  return (
    <ScrollableTabsContext.Provider
      value={{
        activeValue,
        setActiveValue,
        registerContent,
        scrollToSection,
        scrollContainer,
      }}
    >
      {shouldShowTabs ? (
        <div className={cn("relative flex flex-col", className)}>
          {listElement}
          <div
            ref={(node) => {
              if (contentContainerRef) {
                contentContainerRef.current = node;
              }
              measureRef.current = node;
            }}
            className="max-h-[800px] overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300 dark:scrollbar-thumb-zinc-700"
          >
            {contentElements}
          </div>
        </div>
      ) : (
        <div ref={measureRef} className={className}>
          {contentElements}
        </div>
      )}
    </ScrollableTabsContext.Provider>
  );
}

export { ScrollableTabsContent, ScrollableTabsList, ScrollableTabsTrigger };
