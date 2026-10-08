import { useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";

const SCROLL_TO_TOP_THRESHOLD = 200;

interface Args {
  showScrollToTop: boolean;
}

export function useScrollArea({ showScrollToTop }: Args) {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const reduceMotion = useReducedMotion();
  const [isScrolledPastThreshold, setIsScrolledPastThreshold] = useState(false);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!showScrollToTop || !viewport) return;

    function handleScroll() {
      if (!viewport) return;
      setIsScrolledPastThreshold(viewport.scrollTop > SCROLL_TO_TOP_THRESHOLD);
    }

    handleScroll();
    viewport.addEventListener("scroll", handleScroll, { passive: true });
    return () => viewport.removeEventListener("scroll", handleScroll);
  }, [showScrollToTop]);

  function scrollToTop() {
    viewportRef.current?.scrollTo({
      top: 0,
      behavior: reduceMotion ? "auto" : "smooth",
    });
  }

  return { viewportRef, isScrolledPastThreshold, scrollToTop };
}
