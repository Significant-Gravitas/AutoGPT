import { useEffect, useRef, useState } from "react";

// The sentinel sits right after the sticky footer, so it is only on screen
// once the dialog is scrolled to the end; while it is off screen, content is
// scrolling under the footer and the footer shows its divider.
export function useStepFooter() {
  const sentinelRef = useRef<HTMLDivElement>(null);
  const [hasContentBelow, setHasContentBelow] = useState(false);

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(([entry]) => {
      setHasContentBelow(!entry.isIntersecting);
    });
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, []);

  return { sentinelRef, hasContentBelow };
}
