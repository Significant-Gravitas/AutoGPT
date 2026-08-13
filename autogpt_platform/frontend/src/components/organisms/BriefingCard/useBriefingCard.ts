"use client";

import { useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { useCardLayout } from "./useCardLayout";

export function useBriefingCard(rowsKey: string) {
  const listRef = useRef<HTMLUListElement>(null);
  const [isShowingAll, setIsShowingAll] = useState(false);
  const shouldReduceMotion = useReducedMotion();
  const { height, canScrollUp, canScrollDown } = useCardLayout(
    listRef,
    isShowingAll,
    rowsKey,
  );

  // The first measurement necessarily lands one commit after mount, when the
  // card is still at its natural full height. Tweening that commit would play
  // a shrink-to-three-rows animation on every mount, so it snaps instead.
  const hasPaintedRef = useRef(false);
  useEffect(() => {
    hasPaintedRef.current = true;
  }, []);

  function scrollByStep(direction: 1 | -1) {
    const list = listRef.current;
    if (!list) return;
    // Two thirds of the window: the reader keeps a landmark on screen.
    list.scrollBy({
      top: direction * list.clientHeight * 0.66,
      behavior: shouldReduceMotion ? "auto" : "smooth",
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

  const heightTransition =
    shouldReduceMotion || !hasPaintedRef.current
      ? { duration: 0 }
      : { duration: 0.32, ease: [0.32, 0.72, 0, 1] as const };

  return {
    listRef,
    height,
    heightTransition,
    isShowingAll,
    toggleShowAll,
    canScrollUp,
    canScrollDown,
    scrollByStep,
  };
}
