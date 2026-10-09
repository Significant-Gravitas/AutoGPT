import { useSidebar } from "@/components/ui/sidebar";
import { useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { afterAnimations, scrollSidebarTo } from "./helpers";

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
    const heading = recentChatsRef.current?.querySelector<HTMLElement>(
      '[data-sidebar="group-label"]',
    );
    const scrollArea = heading?.closest<HTMLElement>(
      '[data-sidebar="content"]',
    );
    if (!heading || !scrollArea) return;

    heading.focus({ preventScroll: true });
    const behavior = reduceMotion ? "auto" : "smooth";
    // Expanding animates the nav items above Recent chats, and reopening it
    // animates its list open. Measure once they have their final size.
    return afterAnimations(scrollArea, () =>
      scrollSidebarTo(heading, behavior),
    );
  }, [state, reduceMotion]);

  return {
    isRecentChatsOpen,
    setIsRecentChatsOpen,
    recentChatsRef,
    jumpToRecentChats,
  };
}
