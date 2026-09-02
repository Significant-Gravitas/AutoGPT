import { useLayoutEffect, useRef, useState } from "react";

export function useTaskSpec(spec: string) {
  const specRef = useRef<HTMLParagraphElement>(null);
  const [isOverflowing, setIsOverflowing] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useLayoutEffect(() => {
    const el = specRef.current;
    if (!el || expanded) return;

    function measure() {
      if (!el) return;
      setIsOverflowing(el.scrollHeight > el.clientHeight + 1);
    }

    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [expanded, spec]);

  return {
    specRef,
    isOverflowing,
    expanded,
    toggleExpanded: () => setExpanded(!expanded),
  };
}
