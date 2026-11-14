import { useEffect, useRef, useState } from "react";

/** A row that scrolls sideways, and whether some of it is hidden past its
 *  left or right edge. Re-measured on scroll and when the row or its content
 *  resizes. */
export function useOverflowRight<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [hiddenRight, setHiddenRight] = useState(false);
  const [hiddenLeft, setHiddenLeft] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    function measure() {
      if (!el) return;
      setHiddenRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
      setHiddenLeft(el.scrollLeft > 1);
    }
    measure();
    el.addEventListener("scroll", measure, { passive: true });
    const observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(measure);
    observer?.observe(el);
    for (const child of Array.from(el.children)) observer?.observe(child);
    return () => {
      el.removeEventListener("scroll", measure);
      observer?.disconnect();
    };
  }, []);

  return { ref, hiddenRight, hiddenLeft };
}
