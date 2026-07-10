"use client";

import { useSidebar } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import dynamic from "next/dynamic";
import { usePathname } from "next/navigation";
import { parseAsString, useQueryState } from "nuqs";
import { useEffect, useRef, useState } from "react";
import { useCopilotUIStore } from "../../store";
import { useIsMobile } from "../../useIsMobile";

const SandboxIdePanel = dynamic(
  () => import("./SandboxIdePanel").then((m) => m.SandboxIdePanel),
  { ssr: false },
);

const ANIMATION_MS = 300;
// Fixed half-screen width (min == max), per design.
const PANEL_WIDTH = "50vw";

/**
 * Renders the Sandbox IDE panel as a full-height sibling of the layout's
 * `SidebarInset` (new AutoGPT layout only). Slides in/out on open/close and
 * collapses the left sidebar while open. Kept self-contained so
 * `PlatformChrome` stays generic — it no-ops off the copilot route.
 */
export function SandboxIdeSlot() {
  const pathname = usePathname();
  const isMobile = useIsMobile();
  // Child of the new layout: the IDE only shows when both flags are on.
  const isNewLayout = useGetFlag(Flag.AUTOGPT_NEW_LAYOUT);
  const isIdeFlagEnabled = useGetFlag(Flag.AUTOGPT_NEW_LAYOUT_IDE);
  const isEnabled = isNewLayout && isIdeFlagEnabled;
  const [sessionId] = useQueryState("sessionId", parseAsString);
  const isOpen = useCopilotUIStore((s) => s.sandboxIdePanel.isOpen);
  const closeSandboxIdePanel = useCopilotUIStore((s) => s.closeSandboxIdePanel);
  const { open: sidebarOpen, setOpen: setSidebarOpen } = useSidebar();
  const prevSidebarOpen = useRef(sidebarOpen);
  const prevIsOpen = useRef(false);

  const visible =
    !isMobile && pathname === "/copilot" && isEnabled && !!sessionId;

  const [rendered, setRendered] = useState(isOpen);
  const [shown, setShown] = useState(isOpen);
  const [animating, setAnimating] = useState(false);

  // The left sidebar and the sandbox are mutually exclusive: opening one
  // closes the other, but neither force-*reopens* the other — so the user can
  // still collapse the sidebar / close the sandbox and end with both closed.
  //
  // Opening the sandbox ⇒ collapse the left sidebar. Rising-edge only (via a
  // ref) so this fires once when the sandbox opens and never re-closes a
  // sidebar the user just expanded — even if `setOpen`'s identity changes when
  // the sidebar state updates. Initialised to `false` so a default-open
  // sandbox still collapses the sidebar on mount.
  useEffect(() => {
    if (visible && isOpen && !prevIsOpen.current) {
      setSidebarOpen(false);
    }
    prevIsOpen.current = isOpen;
  }, [visible, isOpen, setSidebarOpen]);

  // Expanding the left sidebar ⇒ close the sandbox. Rising-edge guard so the
  // sidebar's default-open state on mount doesn't close a default-open sandbox.
  useEffect(() => {
    if (visible && sidebarOpen && !prevSidebarOpen.current) {
      closeSandboxIdePanel();
    }
    prevSidebarOpen.current = sidebarOpen;
  }, [visible, sidebarOpen, closeSandboxIdePanel]);

  // Drive the enter/exit animation: mount before opening, unmount after closing.
  useEffect(() => {
    setAnimating(true);
    const stopAnimating = setTimeout(() => setAnimating(false), ANIMATION_MS);
    if (isOpen) {
      setRendered(true);
      const raf = requestAnimationFrame(() => setShown(true));
      return () => {
        cancelAnimationFrame(raf);
        clearTimeout(stopAnimating);
      };
    }
    setShown(false);
    const unmount = setTimeout(() => setRendered(false), ANIMATION_MS);
    return () => {
      clearTimeout(unmount);
      clearTimeout(stopAnimating);
    };
  }, [isOpen]);

  if (!visible || !rendered) return null;

  return (
    <div
      className={cn(
        "h-svh shrink-0 overflow-hidden border-l border-l-zinc-100 bg-white",
        animating &&
          "transition-[width] duration-300 ease-out motion-reduce:transition-none",
      )}
      style={{ width: shown ? PANEL_WIDTH : 0 }}
    >
      {/* Inner keeps the content at the fixed width so it doesn't reflow while
          the outer width animates open/closed. */}
      <div className="h-full" style={{ width: PANEL_WIDTH }}>
        <SandboxIdePanel sessionId={sessionId} />
      </div>
    </div>
  );
}
