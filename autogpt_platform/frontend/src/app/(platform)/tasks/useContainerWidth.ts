import { useEffect, useState } from "react";
import { BOARD_MIN_WIDTH } from "./helpers";

export function useContainerWidth() {
  const [element, setElement] = useState<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setWidth(entry.contentRect.width);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [element]);

  return { attach: setElement, isWide: width >= BOARD_MIN_WIDTH };
}
