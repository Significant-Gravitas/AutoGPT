import { useEffect, useRef, useState } from "react";
import {
  centerLastChild,
  padContainerToCenterLastChild,
} from "./conversationScroll";

// A pixel of slack: sub-pixel scroll heights would otherwise leave the bottom
// fade stuck on when the column is already scrolled to the end.
const EDGE_TOLERANCE = 1;
// Smooth centering can take a few frames; ignore scroll events caused by it.
const AUTO_SCROLL_SETTLE_MS = 400;
const SCROLL_KEYS = new Set([
  "ArrowUp",
  "ArrowDown",
  "PageUp",
  "PageDown",
  "Home",
  "End",
  " ",
]);

export function useConversationScroll() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const isFollowingRef = useRef(true);
  const isAutoScrollingRef = useRef(false);
  const autoScrollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [canScrollUp, setCanScrollUp] = useState(false);
  const [canScrollDown, setCanScrollDown] = useState(false);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;

    function distanceFromBottom(target: HTMLDivElement) {
      return target.scrollHeight - target.scrollTop - target.clientHeight;
    }

    function measure() {
      if (!element) return;
      const remaining = distanceFromBottom(element);
      setCanScrollUp(element.scrollTop > EDGE_TOLERANCE);
      setCanScrollDown(remaining > EDGE_TOLERANCE);
    }

    function releaseFollow() {
      isFollowingRef.current = false;
    }

    // A wheel or drag inside an answer field scrolls that field, not the
    // column, so it is not the reader taking over.
    function handlePointerScroll(event: Event) {
      if (isTextEntry(event.target)) return;
      releaseFollow();
    }

    function handleScroll() {
      if (!element) return;
      if (!isAutoScrollingRef.current) {
        // Scrolling back to the end is how the reader re-joins the conversation.
        isFollowingRef.current = distanceFromBottom(element) <= EDGE_TOLERANCE;
      }
      measure();
    }

    function handleKeyDown(event: KeyboardEvent) {
      // Typing an answer bubbles space/arrow keys up here; they move a caret,
      // not the column, so they must not count as taking over the scroll.
      if (isTextEntry(event.target)) return;
      if (SCROLL_KEYS.has(event.key)) {
        releaseFollow();
      }
    }

    function clearAutoScrollGuard() {
      isAutoScrollingRef.current = false;
      if (autoScrollTimerRef.current) {
        clearTimeout(autoScrollTimerRef.current);
        autoScrollTimerRef.current = null;
      }
    }

    function markAutoScrolling(behavior: ScrollBehavior) {
      isAutoScrollingRef.current = true;
      if (autoScrollTimerRef.current) {
        clearTimeout(autoScrollTimerRef.current);
      }
      if (behavior === "smooth") {
        autoScrollTimerRef.current = setTimeout(
          clearAutoScrollGuard,
          AUTO_SCROLL_SETTLE_MS,
        );
      }
    }

    // Steps mount and the typewriter grows text without any scroll or window
    // resize, so content height is what we actually have to watch.
    function handleContentChange() {
      if (!element) return;

      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (!element) return;
          if (isFollowingRef.current) {
            const behavior = prefersReducedMotion() ? "auto" : "smooth";
            markAutoScrolling(behavior);
            centerLastChild(element, behavior);
            if (behavior === "auto") {
              clearAutoScrollGuard();
            }
          } else {
            padContainerToCenterLastChild(element);
          }
          measure();
        });
      });
    }

    measure();
    element.addEventListener("scroll", handleScroll, { passive: true });
    element.addEventListener("wheel", handlePointerScroll, { passive: true });
    element.addEventListener("touchmove", handlePointerScroll, {
      passive: true,
    });
    element.addEventListener("keydown", handleKeyDown);
    element.addEventListener("scrollend", clearAutoScrollGuard, {
      passive: true,
    });

    function observeResizeTargets(root: Element) {
      observer.observe(root);
      Array.from(root.children).forEach((child) => observeResizeTargets(child));
    }

    const observer = new ResizeObserver(handleContentChange);
    observeResizeTargets(element);

    const mutations = new MutationObserver((records) => {
      records.forEach((record) => {
        record.addedNodes.forEach((node) => {
          if (node instanceof Element) observeResizeTargets(node);
        });
      });
      handleContentChange();
    });
    mutations.observe(element, { childList: true });

    return () => {
      element.removeEventListener("scroll", handleScroll);
      element.removeEventListener("wheel", handlePointerScroll);
      element.removeEventListener("touchmove", handlePointerScroll);
      element.removeEventListener("keydown", handleKeyDown);
      element.removeEventListener("scrollend", clearAutoScrollGuard);
      observer.disconnect();
      mutations.disconnect();
      clearAutoScrollGuard();
    };
  }, []);

  return { scrollRef, canScrollUp, canScrollDown };
}

export function isTextEntry(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

function prefersReducedMotion() {
  if (typeof window === "undefined") return true;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
