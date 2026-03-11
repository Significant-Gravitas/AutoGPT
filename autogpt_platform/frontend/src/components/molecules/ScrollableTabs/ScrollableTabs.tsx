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
      <div className={cn("relative flex flex-col", className)}>
        {listElement}
        <div
          ref={(node) => {
            if (contentContainerRef) {
              contentContainerRef.current = node;
            }
          }}
          className="scrollbar-thin max-h-256 overflow-y-auto scrollbar-thumb-zinc-300 scrollbar-track-transparent dark:scrollbar-thumb-zinc-700"
        >
          <div className="min-h-full pb-[200px]">{contentElements}</div>
        </div>
      </div>
    </ScrollableTabsContext.Provider>
  );
}

export { ScrollableTabsContent, ScrollableTabsList, ScrollableTabsTrigger };
