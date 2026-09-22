import { useEffect } from "react";

/**
 * Named wrapper around `useEffect(fn, [])` so the intent ("run on mount
 * and clean up on unmount") is explicit and the exhaustive-deps lint
 * rule doesn't flag a legitimately empty dep array.
 */
export function useMountEffect(effect: () => void | (() => void)): void {
  useEffect(effect, []);
}
