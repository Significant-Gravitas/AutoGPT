import { useEffect, useRef, useState } from "react";

/**
 * Pure state machine for the highlighted-row cursor in a vertical
 * keyboard-driven list. Decoupled from any specific data shape so it
 * can be reused by any list-style organism.
 *
 * - Resets to ``0`` whenever ``resetKey`` changes (e.g. when the search
 *   query changes and the result set is replaced).
 * - Clamps the cursor inside ``[0, totalCount)`` whenever the list
 *   shrinks.
 * - Scrolls the highlighted row into view via ``highlightedRef``.
 */
export function useKeyboardNav(totalCount: number, resetKey: unknown) {
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const highlightedRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    setHighlightedIndex(0);
  }, [resetKey, totalCount]);

  useEffect(() => {
    if (highlightedIndex >= totalCount) {
      setHighlightedIndex(Math.max(0, totalCount - 1));
    }
  }, [highlightedIndex, totalCount]);

  useEffect(() => {
    highlightedRef.current?.scrollIntoView({ block: "nearest" });
  }, [highlightedIndex]);

  function moveHighlight(direction: 1 | -1) {
    if (totalCount === 0) return;
    setHighlightedIndex((current) => {
      const next = current + direction;
      if (next < 0 || next >= totalCount) return current;
      return next;
    });
  }

  return {
    highlightedIndex,
    setHighlightedIndex,
    highlightedRef,
    moveHighlight,
  };
}
