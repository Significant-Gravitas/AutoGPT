"use client";

import { useLayoutEffect, useRef, useState } from "react";

/**
 * Returns one function, created once, that always calls the latest `fn`.
 * Passing it down or through a context no longer changes identity on every
 * render, so memoised consumers can skip their render.
 *
 * The latest `fn` is stored in a layout effect, never during render, so it
 * must only be called from event handlers and effects, not during render.
 */
export function useStableCallback<Args extends unknown[], Result>(
  fn: (...args: Args) => Result,
): (...args: Args) => Result {
  const latest = useRef(fn);
  useLayoutEffect(() => {
    latest.current = fn;
  });
  const [stable] = useState(
    () =>
      (...args: Args) =>
        latest.current(...args),
  );
  return stable;
}
