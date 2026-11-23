import { useEffect, useRef, type RefObject } from "react";

/** When the dialog closes, put focus back on what opened it. Radix only
 *  does that for its own trigger, and the opener here is a plain button
 *  or a menu item. */
export function useReturnFocus(
  isOpen: boolean,
  target: RefObject<HTMLElement | null> | undefined,
) {
  const wasOpen = useRef(isOpen);

  useEffect(() => {
    const closed = wasOpen.current && !isOpen;
    wasOpen.current = isOpen;
    if (!closed || !target) return;
    // After Radix has finished moving focus on close.
    const timer = setTimeout(() => target.current?.focus(), 0);
    return () => clearTimeout(timer);
  }, [isOpen, target]);
}
