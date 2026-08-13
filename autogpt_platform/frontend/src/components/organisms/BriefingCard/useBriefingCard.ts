"use client";

import { useEffect, useRef, useState } from "react";

export const COLLAPSED_ROWS = 3;

// Roughly six rows: enough that "show all" feels like it opened something,
// short enough that the card still sits under the composer.
const MAX_EXPANDED_HEIGHT = 416;

// Sub-pixel scroll offsets are routine (zoom, fractional row heights), so an
// exact comparison would leave both arrows on forever.
const EDGE_TOLERANCE = 2;

// The card animates its real height rather than a `layout` transform: a
// transform-based resize scales the rows' text while it runs and leaves the
// rest of the page unaware that anything moved. Measuring gives framer a
// number to tween, so the column below reflows in step.
export function useBriefingCard() {
  const listRef = useRef<HTMLUListElement>(null);
  const [isShowingAll, setIsShowingAll] = useState(false);
  const [height, setHeight] = useState<number | null>(null);
  const [canScrollUp, setCanScrollUp] = useState(false);
  const [canScrollDown, setCanScrollDown] = useState(false);

  useEffect(() => {
    const list = listRef.current;
    if (!list) return;

    function update() {
      if (!list) return;
      const rows = Array.from(list.children) as HTMLElement[];
      if (rows.length === 0) return;

      if (isShowingAll) {
        setHeight(Math.min(list.scrollHeight, MAX_EXPANDED_HEIGHT));
      } else {
        const lastVisible = rows[Math.min(COLLAPSED_ROWS, rows.length) - 1];
        setHeight(
          lastVisible.offsetTop + lastVisible.offsetHeight - rows[0].offsetTop,
        );
      }

      const { scrollTop, scrollHeight, clientHeight } = list;
      setCanScrollUp(isShowingAll && scrollTop > EDGE_TOLERANCE);
      setCanScrollDown(
        isShowingAll &&
          scrollTop + clientHeight < scrollHeight - EDGE_TOLERANCE,
      );
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
  }, [isShowingAll]);

  function scrollByStep(direction: 1 | -1) {
    const list = listRef.current;
    if (!list) return;
    // Two thirds of the window: the reader keeps a landmark on screen.
    list.scrollBy({
      top: direction * list.clientHeight * 0.66,
      behavior: "smooth",
    });
  }

  function toggleShowAll() {
    // Collapsing keeps whatever scroll offset the reader left behind, so a
    // list read to the middle would shrink around row 7 with no way back up
    // — the three-row window has no scrollbar and no arrows. Jump to the top
    // instantly: a smooth scroll would race the height animation.
    if (isShowingAll) listRef.current?.scrollTo({ top: 0 });
    setIsShowingAll(!isShowingAll);
  }

  return {
    listRef,
    height,
    isShowingAll,
    toggleShowAll,
    canScrollUp,
    canScrollDown,
    scrollByStep,
  };
}
