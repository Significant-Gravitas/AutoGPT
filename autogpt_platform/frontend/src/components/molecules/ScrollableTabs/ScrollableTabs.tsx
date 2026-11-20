"use client";

import { cn } from "@/lib/utils";
import { Children } from "react";
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
      <div
        data-slot="scrollable-tabs"
        className={cn("relative flex flex-col", className)}
      >
        {listElement}
        <div
          ref={(node) => {
            if (contentContainerRef) {
              contentContainerRef.current = node;
            }
          }}
          data-slot="scrollable-tabs-viewport"
          className="scrollbar-thin max-h-[64rem] scrollbar-thumb-border-strong scrollbar-track-transparent overflow-y-auto"
        >
          <div
            data-slot="scrollable-tabs-panels"
            // oxlint-disable-next-line design-system/no-arbitrary-values -- 200px scroll runway below the last panel; the scale has no 200px step
            className="min-h-full pb-[200px]"
          >
            {contentElements}
          </div>
        </div>
      </div>
    </ScrollableTabsContext.Provider>
  );
}

export { ScrollableTabsContent, ScrollableTabsList, ScrollableTabsTrigger };
