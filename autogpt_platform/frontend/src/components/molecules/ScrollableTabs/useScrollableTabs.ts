import { useCallback, useEffect, useRef, useState } from "react";
import { calculateScrollPosition } from "./helpers";

interface Args {
  defaultValue?: string;
}

export function useScrollableTabsInternal({ defaultValue }: Args) {
  const [activeValue, setActiveValue] = useState<string | null>(
    defaultValue || null,
  );
  const contentRefs = useRef<Map<string, HTMLElement>>(new Map());
  const contentContainerRef = useRef<HTMLDivElement | null>(null);
  const scrollContainerRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    scrollContainerRef.current = contentContainerRef.current;
  }, []);

  function registerContent(value: string, element: HTMLElement | null) {
    if (element) {
      contentRefs.current.set(value, element);
    } else {
      contentRefs.current.delete(value);
    }
  }

  function scrollToSection(value: string) {
    const element = contentRefs.current.get(value);
    const scrollContainer = scrollContainerRef.current;
    if (!element) return;

    setActiveValue(value);

    if (scrollContainer) {
      const containerRect = scrollContainer.getBoundingClientRect();
      const elementRect = element.getBoundingClientRect();
      const currentScrollTop = scrollContainer.scrollTop;
      const scrollTop = calculateScrollPosition(
        elementRect,
        containerRect,
        currentScrollTop,
      );

      scrollContainer.scrollTo({
        top: scrollTop,
        behavior: "smooth",
      });
    } else {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  const memoizedRegisterContent = useCallback(registerContent, []);
  const memoizedScrollToSection = useCallback(scrollToSection, []);

  return {
    activeValue,
    setActiveValue,
    registerContent: memoizedRegisterContent,
    scrollToSection: memoizedScrollToSection,
    scrollContainer: scrollContainerRef.current,
    contentContainerRef,
  };
}
