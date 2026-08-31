"use client";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import { MAX_CONTEXT_PANEL_WIDTH, MIN_CONTEXT_PANEL_WIDTH } from "../../store";
import { PanelResizeHandle } from "../PanelResizeHandle";
import { ArtifactsTab } from "./components/ArtifactsTab/ArtifactsTab";
import { useContextPanel } from "./useContextPanel";

interface Props {
  sessionId: string | null;
  mobile?: boolean;
}

// iOS sheet curve — decelerates hard at the end so the chat column settles
// instead of snapping when the panel pushes it aside.
const PANEL_EASE: [number, number, number, number] = [0.32, 0.72, 0, 1];
const PANEL_DURATION = 0.3;

// Files render as a card grid above the composer (``WorkspaceFileCards``), so
// this panel only ever holds the artifacts library. The open flag also drives
// that card, so both the docked desktop panel and the mobile sheet claim the
// screen for the artifacts tab only — a "files" tab belongs to the card.
export function ContextPanel({ sessionId, mobile }: Props) {
  const {
    isOpen,
    activeTab,
    showExpanded,
    closeArtifactPanel,
    contextPanelWidth,
    setContextPanelWidth,
  } = useContextPanel();
  const [isResizing, setIsResizing] = useState(false);
  const shouldReduceMotion = useReducedMotion();

  const library = (
    <div className="flex min-h-0 flex-1 flex-col">
      <ArtifactsTab sessionId={sessionId} />
    </div>
  );

  if (mobile) {
    return (
      <Sheet
        open={isOpen && activeTab === "artifacts"}
        onOpenChange={(open) => !open && closeArtifactPanel()}
      >
        <SheetContent
          side="right"
          className="flex w-full flex-col p-0 sm:max-w-full"
        >
          <SheetHeader className="mt-12 p-2 text-left">
            <SheetTitle className="text-sm font-medium text-zinc-900">
              Artifacts
            </SheetTitle>
          </SheetHeader>
          {library}
        </SheetContent>
      </Sheet>
    );
  }

  const isDocked = showExpanded && activeTab === "artifacts";

  // Width is the animated property because the panel pushes the chat column
  // rather than overlaying it. Dragging the handle bypasses the tween (a
  // queued 300ms tween per pointer move would trail the cursor). The chat's
  // sidebar icon toggles the panel, so it carries no header of its own.
  const transition =
    shouldReduceMotion || isResizing
      ? { duration: 0 }
      : { duration: PANEL_DURATION, ease: PANEL_EASE };

  return (
    <AnimatePresence initial={false}>
      {isDocked && (
        <motion.div
          data-context-panel
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: contextPanelWidth, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={transition}
          className="relative h-full shrink-0 border-l border-l-[#80808017] bg-sidebar"
        >
          {/* Sibling of the clip, not a child of it: the handle is
              -translate-x-1/2 and deliberately straddles the border, so
              clipping it here would halve the drag target to 6px. */}
          <PanelResizeHandle
            panelSelector="[data-context-panel]"
            onWidthChange={setContextPanelWidth}
            onResizingChange={setIsResizing}
            minWidth={MIN_CONTEXT_PANEL_WIDTH}
            maxWidth={MAX_CONTEXT_PANEL_WIDTH}
          />
          <div className="h-full overflow-hidden">
            {/* Fixed inner width: the library keeps its final layout while
                the shell widens, so nothing reflows mid-animation. That also
                means the inner is wider than the shell for the whole tween,
                so the clip above is what stops it spilling over the chat. */}
            <div
              style={{ width: contextPanelWidth }}
              className="flex h-full min-h-0 flex-col overflow-hidden"
            >
              {library}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
