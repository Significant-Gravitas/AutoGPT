import { RefObject, useEffect, useState } from "react";

const BOTTOM_THRESHOLD_PX = 2;

export function useBottomScrollShadow(ref: RefObject<HTMLElement | null>) {
  const [hasMoreBelow, setHasMoreBelow] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    function update() {
      if (!element) return;
      const remaining =
        element.scrollHeight - element.scrollTop - element.clientHeight;
      setHasMoreBelow(remaining > BOTTOM_THRESHOLD_PX);
    }

    update();
    element.addEventListener("scroll", update, { passive: true });
    const observer =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(update);
    observer?.observe(element);
    Array.from(element.children).forEach((child) => observer?.observe(child));

    return () => {
      element.removeEventListener("scroll", update);
      observer?.disconnect();
    };
  }, [ref]);

  return hasMoreBelow;
}
