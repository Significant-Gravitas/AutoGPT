import { useSidebar } from "@/components/ui/sidebar";
import { useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { afterAnimations, scrollSidebarToWhenReachable } from "./helpers";

export function useJumpToNavGroup() {
  const { state, setOpen } = useSidebar();
  const reduceMotion = useReducedMotion();
  const [isOpen, setIsOpen] = useState(true);
  const groupRef = useRef<HTMLDivElement>(null);
  const isJumpPendingRef = useRef(false);

  function jumpToGroup() {
    isJumpPendingRef.current = true;
    setIsOpen(true);
    setOpen(true);
  }

  useEffect(() => {
    if (state !== "expanded" || !isJumpPendingRef.current) return;
    isJumpPendingRef.current = false;
    const group = groupRef.current;
    const heading = group?.querySelector<HTMLElement>(
      '[data-sidebar="group-label"]',
    );
    const scrollArea = heading?.closest<HTMLElement>(
      '[data-sidebar="content"]',
    );
    if (!group || !heading || !scrollArea) return;

    heading.focus({ preventScroll: true });
    const behavior = reduceMotion ? "auto" : "smooth";
    let stopFollowing = () => {};
    // Expanding animates the nav items above the group, and reopening it
    // animates its list open (and may reload it). Measure once they have
    // their final size, and follow the list while it is still loading.
    const cancelWaiting = afterAnimations(scrollArea, () => {
      stopFollowing = scrollSidebarToWhenReachable(heading, group, behavior);
    });
    return () => {
      cancelWaiting();
      stopFollowing();
    };
  }, [state, reduceMotion]);

  return {
    isOpen,
    setIsOpen,
    groupRef,
    jumpToGroup,
  };
}
