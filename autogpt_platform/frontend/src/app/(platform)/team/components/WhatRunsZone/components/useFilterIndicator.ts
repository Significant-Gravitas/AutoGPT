import { useLayoutEffect, useRef, useState } from "react";

interface Indicator {
  left: number;
  width: number;
}

// The pill is one element that slides between chips rather than a background
// on each chip, so switching tabs reads as a single travelling object. Measured
// rather than CSS-only because the chips are text-width, not a fixed grid.
export function useFilterIndicator(activeId: string) {
  const listRef = useRef<HTMLDivElement>(null);
  const [indicator, setIndicator] = useState<Indicator | null>(null);

  useLayoutEffect(() => {
    const list = listRef.current;
    if (!list) return;

    function measure() {
      const list = listRef.current;
      const active = list?.querySelector<HTMLElement>('[data-active="true"]');
      if (!list || !active) return;
      setIndicator({ left: active.offsetLeft, width: active.offsetWidth });
    }

    measure();
    // Chips wrap on narrow viewports, so the pill has to re-measure on resize.
    const observer = new ResizeObserver(measure);
    observer.observe(list);
    return () => observer.disconnect();
  }, [activeId]);

  return { listRef, indicator };
}
