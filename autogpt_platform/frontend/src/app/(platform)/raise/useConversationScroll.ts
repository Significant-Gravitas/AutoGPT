import { useEffect, useRef, useState } from "react";

// A pixel of slack: sub-pixel scroll heights would otherwise leave the bottom
// fade stuck on when the column is already scrolled to the end.
const EDGE_TOLERANCE = 1;
// Scrolling up by more than this releases the auto-follow, so reading back
// through the conversation is not fought by every new message.
const STICK_THRESHOLD = 64;

export function useConversationScroll() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const isPinnedRef = useRef(true);
  const [canScrollUp, setCanScrollUp] = useState(false);
  const [canScrollDown, setCanScrollDown] = useState(false);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;

    function distanceFromBottom(target: HTMLDivElement) {
      return target.scrollHeight - target.scrollTop - target.clientHeight;
    }

    function measure() {
      if (!element) return;
      const remaining = distanceFromBottom(element);
      setCanScrollUp(element.scrollTop > EDGE_TOLERANCE);
      setCanScrollDown(remaining > EDGE_TOLERANCE);
    }

    function handleScroll() {
      if (!element) return;
      isPinnedRef.current = distanceFromBottom(element) <= STICK_THRESHOLD;
      measure();
    }

    // Steps mount and the typewriter grows text without any scroll or window
    // resize, so content height is what we actually have to watch.
    function handleContentChange() {
      if (!element) return;
      if (isPinnedRef.current) {
        element.scrollTo({
          top: element.scrollHeight,
          behavior: prefersReducedMotion() ? "auto" : "smooth",
        });
      }
      measure();
    }

    measure();
    element.addEventListener("scroll", handleScroll, { passive: true });
    const observer = new ResizeObserver(handleContentChange);
    observer.observe(element);
    Array.from(element.children).forEach((child) => observer.observe(child));

    const mutations = new MutationObserver((records) => {
      records.forEach((record) => {
        record.addedNodes.forEach((node) => {
          if (node instanceof Element) observer.observe(node);
        });
      });
      handleContentChange();
    });
    mutations.observe(element, { childList: true });

    return () => {
      element.removeEventListener("scroll", handleScroll);
      observer.disconnect();
      mutations.disconnect();
    };
  }, []);

  return { scrollRef, canScrollUp, canScrollDown };
}

function prefersReducedMotion() {
  if (typeof window === "undefined") return true;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
