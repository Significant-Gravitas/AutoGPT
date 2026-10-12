import { useSidebar } from "@/components/ui/sidebar";
import { useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { afterAnimations, scrollSidebarToWhenReachable } from "./helpers";

export function useJumpToRecentChats() {
  const { state, setOpen } = useSidebar();
  const reduceMotion = useReducedMotion();
  const [isRecentChatsOpen, setIsRecentChatsOpen] = useState(true);
  const recentChatsRef = useRef<HTMLDivElement>(null);
  const isJumpPendingRef = useRef(false);

  function jumpToRecentChats() {
    isJumpPendingRef.current = true;
    setIsRecentChatsOpen(true);
    setOpen(true);
  }

  useEffect(() => {
    if (state !== "expanded" || !isJumpPendingRef.current) return;
    isJumpPendingRef.current = false;
    const recentChats = recentChatsRef.current;
    const heading = recentChats?.querySelector<HTMLElement>(
      '[data-sidebar="group-label"]',
    );
    const scrollArea = heading?.closest<HTMLElement>(
      '[data-sidebar="content"]',
    );
    if (!recentChats || !heading || !scrollArea) return;

    heading.focus({ preventScroll: true });
    const behavior = reduceMotion ? "auto" : "smooth";
    let stopFollowing = () => {};
    // Expanding animates the nav items above Recent chats, and reopening it
    // animates its list open (and may reload it). Measure once they have
    // their final size, and follow the list while it is still loading.
    const cancelWaiting = afterAnimations(scrollArea, () => {
      stopFollowing = scrollSidebarToWhenReachable(
        heading,
        recentChats,
        behavior,
      );
    });
    return () => {
      cancelWaiting();
      stopFollowing();
    };
  }, [state, reduceMotion]);

  return {
    isRecentChatsOpen,
    setIsRecentChatsOpen,
    recentChatsRef,
    jumpToRecentChats,
  };
}
