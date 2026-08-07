import { useRef, useState } from "react";

/** Measures whether a clamped element actually overflows, instead of
 *  guessing from character count — a short bio can still clamp on narrow or
 *  zoomed layouts. Re-measures on element resize. */
export function useClampOverflow() {
  const [isOverflowing, setIsOverflowing] = useState(false);
  const observerRef = useRef<ResizeObserver | null>(null);

  function measureRef(node: HTMLElement | null) {
    observerRef.current?.disconnect();
    observerRef.current = null;
    if (!node) return;
    function measure() {
      if (!node) return;
      setIsOverflowing(node.scrollHeight > node.clientHeight + 1);
    }
    if (typeof ResizeObserver !== "undefined") {
      observerRef.current = new ResizeObserver(measure);
      observerRef.current.observe(node);
    }
    measure();
  }

  return { isOverflowing, measureRef };
}
