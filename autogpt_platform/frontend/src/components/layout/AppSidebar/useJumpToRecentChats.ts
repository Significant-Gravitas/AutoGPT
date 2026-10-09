import { useSidebar } from "@/components/ui/sidebar";
import { useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { scrollSidebarTo, whenHeightSettles } from "./helpers";

type PendingJump = "now" | "afterOpening";

export function useJumpToRecentChats() {
  const { state, setOpen } = useSidebar();
  const reduceMotion = useReducedMotion();
  const [isRecentChatsOpen, setIsRecentChatsOpen] = useState(true);
  const recentChatsRef = useRef<HTMLDivElement>(null);
  const pendingJumpRef = useRef<PendingJump | null>(null);

  function jumpToRecentChats() {
    pendingJumpRef.current = isRecentChatsOpen ? "now" : "afterOpening";
    setIsRecentChatsOpen(true);
    setOpen(true);
  }

  useEffect(() => {
    if (state !== "expanded") return;
    const jump = pendingJumpRef.current;
    pendingJumpRef.current = null;
    const recentChats = recentChatsRef.current;
    const heading = recentChats?.querySelector<HTMLElement>(
      '[data-sidebar="group-label"]',
    );
    if (!jump || !recentChats || !heading) return;

    heading.focus({ preventScroll: true });
    const behavior = reduceMotion ? "auto" : "smooth";
    if (jump === "now") {
      scrollSidebarTo(heading, behavior);
      return;
    }
    // Reopening Recent chats animates its list open, and the heading can't
    // reach the top until the list below it has its full height.
    return whenHeightSettles(recentChats, () =>
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
