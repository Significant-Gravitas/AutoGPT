import { useEffect, useState, type RefObject } from "react";

// Whether a scroll container has content hidden above or below its viewport.
// Re-measured on scroll and on resize (the grid reflows across breakpoints),
// so a container that fits entirely reports neither edge as hidden.
export function useScrollEdges(ref: RefObject<HTMLElement | null>) {
  const [edges, setEdges] = useState({
    hiddenAbove: false,
    hiddenBelow: false,
  });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    function measure() {
      if (!el) return;
      const hiddenAbove = el.scrollTop > 1;
      const hiddenBelow = el.scrollTop + el.clientHeight < el.scrollHeight - 1;
      setEdges((prev) =>
        prev.hiddenAbove === hiddenAbove && prev.hiddenBelow === hiddenBelow
          ? prev
          : { hiddenAbove, hiddenBelow },
      );
    }

    measure();
    el.addEventListener("scroll", measure, { passive: true });
    const observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(measure);
    observer?.observe(el);
    return () => {
      el.removeEventListener("scroll", measure);
      observer?.disconnect();
    };
  }, [ref]);

  return edges;
}
