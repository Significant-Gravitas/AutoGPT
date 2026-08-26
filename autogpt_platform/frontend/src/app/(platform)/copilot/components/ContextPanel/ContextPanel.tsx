"use client";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import {
  MAX_CONTEXT_PANEL_WIDTH,
  MIN_CONTEXT_PANEL_WIDTH,
  type ContextPanelTab,
} from "../../store";
import { PanelResizeHandle } from "../PanelResizeHandle";
import { ArtifactsTab } from "./components/ArtifactsTab/ArtifactsTab";
import { FilesTab } from "./components/FilesTab/FilesTab";
import { useSessionFiles } from "./components/FilesTab/useSessionFiles";
import { ProgressTab } from "./components/ProgressTab/ProgressTab";
import { TabSwitcher } from "./components/TabSwitcher";
import { useContextPanel } from "./useContextPanel";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  sessionId: string | null;
  mobile?: boolean;
}

// iOS sheet curve — decelerates hard at the end so the chat column settles
// instead of snapping when the panel pushes it aside.
const PANEL_EASE: [number, number, number, number] = [0.32, 0.72, 0, 1];
const PANEL_DURATION = 0.3;

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
  const { uploaded, generated } = useSessionFiles(sessionId);
  const filesCount = uploaded.length + generated.length;
  // New tool UI renders files as a card grid above the composer
  // (``WorkspaceFileCards``), so the desktop panel only ever docks for the
  // artifacts library there — files and progress tabs stay out of it.
  const isNewToolUI = useGetFlag(Flag.NEW_TOOL_UI);
  // When the task bar (above the chat input) is on, the sidebar drops the
  // Progress tab and shows Files only.
  const showProgressTab = !useGetFlag(Flag.TASK_PROGRESS_BAR) && !isNewToolUI;
  const showFilesTab = !isNewToolUI;
  // Clamp a persisted tab to one that still exists, so the switcher never ends
  // up with no selected tab. The Artifacts library tab ships with the new
  // tool UI only.
  const availableTabs: ContextPanelTab[] = [
    ...(showProgressTab ? (["progress"] as const) : []),
    ...(showFilesTab ? (["files"] as const) : []),
    ...(isNewToolUI ? (["artifacts"] as const) : []),
  ];
  // Read-side only: "files" stays "files" in the store because the toggle,
  // the workspace-files card and the chat column all read the raw tab.
  const effectiveTab = availableTabs.includes(activeTab)
    ? activeTab
    : availableTabs[0];

  // One tab left (new tool UI docks the panel for artifacts only) means the
  // switcher has nothing to switch, and the chat's sidebar icon already
  // toggles the panel — so the header row has nothing left to hold. The old
  // UI keeps its header regardless: it carries the panel's close button.
  const showHeader = mobile || !isNewToolUI || availableTabs.length > 1;

  const tabs = (
    <div className="flex min-h-0 flex-1 flex-col">
      {showHeader && (
        <div
          className={cn(
            "flex items-center justify-between gap-2 p-2",
            mobile && "mt-12",
          )}
        >
          <TabSwitcher
            activeTab={effectiveTab}
            filesCount={filesCount}
            onChange={setActiveTab}
            showProgressTab={showProgressTab}
            showFilesTab={showFilesTab}
            showArtifactsTab={isNewToolUI}
          />
          {!mobile && !isNewToolUI && (
            <button
              type="button"
              onClick={() => closeArtifactPanel()}
              title="Close"
              aria-label="Close workspace panel"
              className="rounded p-1.5 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
            >
              <Icon icon={Cancel01Icon} size={16} />
            </button>
          )}
        </div>
      )}
      <div className="flex min-h-0 flex-1 flex-col">
        {effectiveTab === "progress" ? (
          <ProgressTab sessionId={sessionId} />
        ) : effectiveTab === "artifacts" ? (
          <ArtifactsTab sessionId={sessionId} />
        ) : (
          <FilesTab sessionId={sessionId} />
        )}
      </div>
    </div>
  );

  if (mobile) {
    return (
      <Sheet
        open={isOpen}
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

  // Under the new tool UI the open flag also drives the inline files card, so
  // the docked panel only claims the side region for the artifacts tab. The
  // raw tab decides: a "files" tab belongs to the card, not to this panel.
  const isDocked = showExpanded && !(isNewToolUI && activeTab !== "artifacts");

  // Width is the animated property because the panel pushes the chat column
  // rather than overlaying it. Dragging the handle bypasses the tween (a
  // queued 300ms tween per pointer move would trail the cursor). The old UI
  // always mounts instantly — the open/close tween is new-tool-UI only.
  const transition =
    shouldReduceMotion || isResizing || !isNewToolUI
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
