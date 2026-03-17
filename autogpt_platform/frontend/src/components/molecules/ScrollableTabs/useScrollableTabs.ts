import { useCallback, useRef, useState } from "react";
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

  function registerContent(value: string, element: HTMLElement | null) {
    if (element) {
      contentRefs.current.set(value, element);
    } else {
      contentRefs.current.delete(value);
    }
  }

  function scrollToSection(value: string) {
    const element = contentRefs.current.get(value);
    const scrollContainer = contentContainerRef.current;
    if (!element || !scrollContainer) return;

    setActiveValue(value);

    const containerRect = scrollContainer.getBoundingClientRect();
    const elementRect = element.getBoundingClientRect();
    const currentScrollTop = scrollContainer.scrollTop;
    const scrollTop = calculateScrollPosition(
      elementRect,
      containerRect,
      currentScrollTop,
    );

    const maxScrollTop =
      scrollContainer.scrollHeight - scrollContainer.clientHeight;
    const clampedScrollTop = Math.min(Math.max(0, scrollTop), maxScrollTop);

    scrollContainer.scrollTo({
      top: clampedScrollTop,
      behavior: "smooth",
    });
  }

  const memoizedRegisterContent = useCallback(registerContent, []);
  const memoizedScrollToSection = useCallback(scrollToSection, []);

  return {
    activeValue,
    setActiveValue,
    registerContent: memoizedRegisterContent,
    scrollToSection: memoizedScrollToSection,
    scrollContainer: contentContainerRef.current,
    contentContainerRef,
  };
}
