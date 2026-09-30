import { useEffect, useRef, useState } from "react";

/** Matches the duration the exit transition is given in {@link Swap}. */
const SWAP_MS = 150;

function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Holds content one beat behind its value so a change can be animated.
 *
 * The old content leaves upward and the replacement arrives from below, which
 * reads as the value being swapped rather than the same spot rewriting itself
 * in place. That needs the outgoing content on screen after the new one has
 * arrived, so what is displayed is state rather than the prop.
 *
 * Keyed by a string because the content may be a node, and nodes cannot be
 * compared: the key is what says a swap has happened.
 */
export function useSwap<T>(key: string, value: T) {
  const [shownKey, setShownKey] = useState(key);
  const [shown, setShown] = useState(value);
  const [phase, setPhase] = useState<"idle" | "exit" | "enter">("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (key === shownKey) {
      // Back to what is already on screen while its own exit was still
      // running: a second click inside SWAP_MS. The timer was cleared on the
      // way in, so nothing is left to bring the content back, and an exit left
      // standing holds it at opacity 0 until some later swap happens to clear
      // it. There is no longer anything to animate, so it simply stays.
      setPhase((current) => (current === "exit" ? "idle" : current));
      return;
    }
    if (prefersReducedMotion()) {
      setShownKey(key);
      setShown(value);
      return;
    }
    setPhase("exit");
    timer.current = setTimeout(() => {
      setShownKey(key);
      setShown(value);
      setPhase("enter");
    }, SWAP_MS);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [key, shownKey, value]);

  useEffect(() => {
    if (phase !== "enter") return;
    // The entry position has to be painted before the transition back to rest
    // is allowed, or the browser collapses both into a single frame and there
    // is nothing to see. Two frames is what guarantees that paint.
    let inner = 0;
    const outer = requestAnimationFrame(() => {
      inner = requestAnimationFrame(() => setPhase("idle"));
    });
    return () => {
      cancelAnimationFrame(outer);
      cancelAnimationFrame(inner);
    };
  }, [phase]);

  return { shown, phase };
}
