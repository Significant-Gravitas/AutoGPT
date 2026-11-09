"use client";

import { useInsertionEffect, useRef, useState } from "react";

/**
 * Returns one function, created once, that always calls the latest `fn`.
 * Passing it down or through a context no longer changes identity on every
 * render, so memoised consumers can skip their render.
 *
 * The latest `fn` is stored in an insertion effect, never during render. That
 * runs before every layout effect of the same commit, so handlers and effects
 * (layout ones in children included) already see it; calling it during render
 * is not supported.
 */
export function useStableCallback<Args extends unknown[], Result>(
  fn: (...args: Args) => Result,
): (...args: Args) => Result {
  const latest = useRef(fn);
  useInsertionEffect(() => {
    latest.current = fn;
  });
  const [stable] = useState(
    () =>
      (...args: Args) =>
        latest.current(...args),
  );
  return stable;
}
