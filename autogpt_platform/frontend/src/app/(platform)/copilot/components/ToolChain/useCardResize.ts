"use client";

import { useLayoutEffect, useRef, useState } from "react";

/** Measures the card body so its clipping wrapper can tween between the
 *  collapsed and expanded heights — `height: auto` is not interpolable, so
 *  the transition needs a real pixel value on both sides of the change. */
export function useCardResize(expanded: boolean) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number>();

  useLayoutEffect(() => {
    const el = contentRef.current;
    if (!el) return;
    function measure() {
      if (el) setHeight(el.getBoundingClientRect().height);
    }
    measure();
    // Text reflow (window resize, font swap) changes the clamped height too.
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [expanded]);

  return { contentRef, height };
}
