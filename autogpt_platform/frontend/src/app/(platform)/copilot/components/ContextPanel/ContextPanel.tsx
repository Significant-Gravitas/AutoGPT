"use client";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useEffect, useState } from "react";
import {
  MAX_CONTEXT_PANEL_WIDTH,
  MIN_CONTEXT_PANEL_WIDTH,
  useCopilotUIStore,
} from "../../store";
import { PanelResizeHandle } from "../PanelResizeHandle";
import { ContextPanelPopover } from "./ContextPanelPopover";
import { FilesTab } from "./components/FilesTab/FilesTab";
import { useSessionFiles } from "./components/FilesTab/useSessionFiles";
import { ProgressTab } from "./components/ProgressTab/ProgressTab";
import { TabSwitcher } from "./components/TabSwitcher";
import { useContextPanel } from "./useContextPanel";

// Open/close animation duration — the width transition (which pushes the chat
// area) and the mount/unmount timers share this so they stay in lockstep.
const PANEL_ANIM_MS = 350;

interface Props {
  sessionId: string | null;
  mobile?: boolean;
}

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
  const isSandboxOpen = useCopilotUIStore((s) => s.sandboxIdePanel.isOpen);
  const { uploaded, generated } = useSessionFiles(sessionId);

  // Drive the slide-in/out: mount before opening, unmount after closing, and
  // flag `animating` only during the transition so live drag-resizing (which
  // also changes the width) stays instant instead of easing.
  const [rendered, setRendered] = useState(showExpanded);
  const [shown, setShown] = useState(showExpanded);
  const [animating, setAnimating] = useState(false);

  useEffect(() => {
    setAnimating(true);
    const stopAnimating = setTimeout(() => setAnimating(false), PANEL_ANIM_MS);
    if (showExpanded) {
      setRendered(true);
      const raf = requestAnimationFrame(() => setShown(true));
      return () => {
        cancelAnimationFrame(raf);
        clearTimeout(stopAnimating);
      };
    }
    setShown(false);
    const unmount = setTimeout(() => setRendered(false), PANEL_ANIM_MS);
    return () => {
      clearTimeout(unmount);
      clearTimeout(stopAnimating);
    };
  }, [showExpanded]);
  const filesCount = uploaded.length + generated.length;
  // When the task bar (above the chat input) is on, the sidebar drops the
  // Progress tab and shows Files only.
  const showProgressTab = !useGetFlag(Flag.TASK_PROGRESS_BAR);
  // Clamp a persisted "progress" tab to "files" when Progress is hidden, so
  // the switcher never ends up with no selected tab.
  const effectiveTab = showProgressTab ? activeTab : "files";

  // The tab switcher only earns its header row when there's more than one tab
  // (Progress + Files). With the task-progress bar on, only Files remains, so
  // the row is dropped — the header icon toggles the card instead.
  const showTabHeader = showProgressTab || mobile;

  const tabs = (
    <div className="flex min-h-0 flex-1 flex-col">
      {showTabHeader && (
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
          />
        </div>
      )}
      <div className="flex min-h-0 flex-1 flex-col">
        {showProgressTab && effectiveTab === "progress" ? (
          <ProgressTab sessionId={sessionId} />
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

  // While the sandbox IDE owns the right side, there's no room for the inline
  // panel — present the files card as a floating popover over the chat instead.
  if (isSandboxOpen) {
    return (
      <ContextPanelPopover open={showExpanded} onClose={closeArtifactPanel}>
        {tabs}
      </ContextPanelPopover>
    );
  }

  if (!rendered) return null;

  return (
    <div
      data-context-panel
      style={{ width: shown ? contextPanelWidth : 0, willChange: "width" }}
      className={cn(
        "relative my-2 mr-2 flex max-h-[calc(100%-1rem)] shrink-0 flex-col self-start overflow-hidden rounded-[2rem] border border-zinc-100 bg-white shadow-sm [corner-shape:squircle]",
        animating &&
          "duration-[350ms] ease-[cubic-bezier(0.32,0.72,0,1)] transition-[width] motion-reduce:transition-none",
      )}
    >
      <PanelResizeHandle
        panelSelector="[data-context-panel]"
        onWidthChange={setContextPanelWidth}
        minWidth={MIN_CONTEXT_PANEL_WIDTH}
        maxWidth={MAX_CONTEXT_PANEL_WIDTH}
      />
      {/* Fixed-width inner so the files content doesn't reflow while the outer
          width animates open/closed — it slides in as the width is revealed. */}
      <div
        style={{ width: contextPanelWidth }}
        className="flex min-h-0 flex-1 flex-col overflow-hidden"
      >
        {tabs}
      </div>
    </div>
  );
}
