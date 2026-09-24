import { useEffect, useState } from "react";

// True while the queue's place in the chat is below the visible area, so it
// can stay reachable as a header without covering the reply being read.
export function useStuckToBottom() {
  const [sentinel, sentinelRef] = useState<HTMLDivElement | null>(null);
  const [stuck, setStuck] = useState(false);

  useEffect(() => {
    if (!sentinel || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(([entry]) => {
      const bottom = entry.rootBounds?.bottom ?? window.innerHeight;
      setStuck(!entry.isIntersecting && entry.boundingClientRect.top > bottom);
    });
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [sentinel]);

  function expand() {
    sentinel?.scrollIntoView({ block: "start", behavior: "smooth" });
  }

  return { sentinelRef, stuck, expand };
}
