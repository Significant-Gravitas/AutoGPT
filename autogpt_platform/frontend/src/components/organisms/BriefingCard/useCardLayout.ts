"use client";

import { useLayoutEffect, useState, type RefObject } from "react";
import { getScrollEdges, measureListHeight } from "./helpers";

// The card animates its real height rather than a `layout` transform: a
// transform-based resize scales the rows' text while it runs and leaves the
// rest of the page unaware that anything moved. Measuring gives framer a
// number to tween, so the column below reflows in step.
//
// `rowsKey` identifies the rendered rows. The briefing query refetches on
// focus and reconnect, and a refetch that swaps the run list would leave the
// observer bound to detached rows — the card would keep the height it
// measured for the old ones until the reader toggled it.
export function useCardLayout(
  listRef: RefObject<HTMLUListElement>,
  isShowingAll: boolean,
  rowsKey: string,
) {
  const [height, setHeight] = useState<number | null>(null);
  const [canScrollUp, setCanScrollUp] = useState(false);
  const [canScrollDown, setCanScrollDown] = useState(false);

  // Layout effect, not effect: measuring after paint lets the browser show
  // the card at full height for a frame before it collapses.
  useLayoutEffect(() => {
    const list = listRef.current;
    if (!list) return;

    function update() {
      if (!list) return;
      const measured = measureListHeight(list, isShowingAll);
      if (measured === null) return;
      setHeight(measured);

      const edges = getScrollEdges(list);
      setCanScrollUp(isShowingAll && edges.canScrollUp);
      setCanScrollDown(isShowingAll && edges.canScrollDown);
    }

    update();
    list.addEventListener("scroll", update, { passive: true });
    // Avatars and wrapped summaries settle after the first paint, which
    // changes the height the card should animate to.
    const observer = new ResizeObserver(update);
    observer.observe(list);
    for (const row of Array.from(list.children)) observer.observe(row);

    return () => {
      list.removeEventListener("scroll", update);
      observer.disconnect();
    };
  }, [listRef, isShowingAll, rowsKey]);

  return { height, canScrollUp, canScrollDown };
}
