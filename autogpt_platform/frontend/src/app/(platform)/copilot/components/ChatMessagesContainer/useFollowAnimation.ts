import { useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import type { Animation, SpringAnimation } from "use-stick-to-bottom";

/** How the view follows a growing thread while the reader is at its end: a
 *  soft spring, so a streamed line or a tall card that mounts (a tool card,
 *  a code block, an image) glides into view instead of jolting it. */
export const FOLLOW_SPRING: SpringAnimation = {
  damping: 0.7,
  stiffness: 0.05,
  mass: 1.25,
};
/** Growth right after a thread opens (rows measuring, code highlighting,
 *  images) still pins it to its end at once, so it lands there. */
export const LANDING_GRACE_MS = 1_000;
/** A width change (a side panel opening, a window resize) reflows every row
 *  together; the view stays pinned through it rather than drifting. */
export const REFLOW_HOLD_MS = 400;

interface Args {
  /** The thread is on screen: its first frame has scrolled to the end. */
  settled: boolean;
  sessionID: string | null | undefined;
}

/**
 * The `resize` animation for the thread's StickToBottom. Instant until the
 * thread has landed, while it reflows to a new width, and whenever the
 * reader prefers reduced motion; a spring otherwise. Either way StickToBottom
 * only follows a reader who is at the end: one who scrolled up stays put.
 * `widthRef` goes on an element as wide as the thread.
 */
export function useFollowAnimation({ settled, sessionID }: Args) {
  const reduceMotion = useReducedMotion();
  const thread = sessionID ?? null;
  const [landedThread, setLandedThread] = useState<string | null>(null);
  const [isReflowing, setIsReflowing] = useState(false);
  const widthRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!settled) return;
    const timer = setTimeout(() => setLandedThread(thread), LANDING_GRACE_MS);
    return () => clearTimeout(timer);
  }, [settled, thread]);

  useEffect(() => {
    const probe = widthRef.current;
    if (!probe || typeof ResizeObserver === "undefined") return;
    let width: number | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const observer = new ResizeObserver(([entry]) => {
      const next = entry.contentRect.width;
      const changed = width !== null && next !== width;
      width = next;
      if (!changed) return;
      setIsReflowing(true);
      clearTimeout(timer);
      timer = setTimeout(() => setIsReflowing(false), REFLOW_HOLD_MS);
    });
    observer.observe(probe);
    return () => {
      observer.disconnect();
      clearTimeout(timer);
    };
    // The thread's StickToBottom remounts per session, and the probe with it.
  }, [thread]);

  const smooth =
    settled && landedThread === thread && !isReflowing && !reduceMotion;
  const resize: Animation = smooth ? FOLLOW_SPRING : "instant";
  return { resize, widthRef };
}
