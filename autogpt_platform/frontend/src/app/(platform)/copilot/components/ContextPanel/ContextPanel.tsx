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
import { TabSwitcher } from "./components/TabSwitcher";
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
// this panel only ever docks for the artifacts library.
export function ContextPanel({ sessionId, mobile }: Props) {
  const {
    isOpen,
    activeTab,
    showExpanded,
    setActiveTab,
    closeArtifactPanel,
    contextPanelWidth,
    setContextPanelWidth,
  } = useContextPanel();
  const [isResizing, setIsResizing] = useState(false);
  const shouldReduceMotion = useReducedMotion();

  // On desktop there is a single tab and the chat's sidebar icon already
  // toggles the panel, so the header row has nothing to hold; the mobile
  // sheet keeps it as a title.
  const tabs = (
    <div className="flex min-h-0 flex-1 flex-col">
      {mobile && (
        <div className="mt-12 flex items-center justify-between gap-2 p-2">
          <TabSwitcher activeTab="artifacts" onChange={setActiveTab} />
        </div>
      )}
      <div className="flex min-h-0 flex-1 flex-col">
        <ArtifactsTab sessionId={sessionId} />
      </div>
    </div>
  );

  // The open flag also drives the inline files card, so the sheet — like the
  // docked desktop panel below — only claims the screen for the artifacts
  // tab. A "files" tab belongs to the card, not to this sheet.
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
          <SheetHeader className="sr-only">
            <SheetTitle>Workspace</SheetTitle>
          </SheetHeader>
          {tabs}
        </SheetContent>
      </Sheet>
    );
  }

  // The open flag also drives the inline files card, so the docked panel only
  // claims the side region for the artifacts tab. The raw tab decides: a
  // "files" tab belongs to the card, not to this panel.
  const isDocked = showExpanded && activeTab === "artifacts";

  // Width is the animated property because the panel pushes the chat column
  // rather than overlaying it. Dragging the handle bypasses the tween (a
  // queued 300ms tween per pointer move would trail the cursor).
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
            {/* Fixed inner width: the tabs keep their final layout while the
                shell widens, so nothing reflows mid-animation. That also
                means the inner is wider than the shell for the whole tween,
                so the clip above is what stops it spilling over the chat. */}
            <div
              style={{ width: contextPanelWidth }}
              className="flex h-full min-h-0 flex-col overflow-hidden"
            >
              {tabs}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
