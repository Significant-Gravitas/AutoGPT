"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import { XIcon } from "@phosphor-icons/react";
import { AnimatePresence, motion } from "framer-motion";
import { ArtifactContent } from "../ArtifactPanel/components/ArtifactContent";
import { ArtifactDragHandle } from "../ArtifactPanel/components/ArtifactDragHandle";
import { ArtifactMinimizedStrip } from "../ArtifactPanel/components/ArtifactMinimizedStrip";
import { ArtifactPanelHeader } from "../ArtifactPanel/components/ArtifactPanelHeader";
import { useArtifactPanel } from "../ArtifactPanel/useArtifactPanel";
import { ArtifactsTab } from "./components/ArtifactsTab";
import { FilesTab } from "./components/FilesTab/FilesTab";
import { ProgressTab } from "./components/ProgressTab";
import { TabSwitcher } from "./components/TabSwitcher";
import { useContextPanel } from "./useContextPanel";

interface Props {
  sessionId: string | null;
  mobile?: boolean;
}

export function ContextPanel({ sessionId, mobile }: Props) {
  const {
    isOpen,
    activeTab,
    view,
    setActiveTab,
    setArtifactPanelWidth,
    closeArtifactPanel,
  } = useContextPanel();
  const preview = useArtifactPanel();

  function renderTabs() {
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        {!mobile && (
          <div className="flex justify-end p-2">
            <Button
              variant="ghost"
              size="small"
              onClick={closeArtifactPanel}
              aria-label="Close workspace panel"
            >
              <XIcon size={16} />
            </Button>
          </div>
        )}
        <div className="p-2">
          <TabSwitcher activeTab={activeTab} onChange={setActiveTab} />
        </div>
        {activeTab === "progress" && <ProgressTab />}
        {activeTab === "files" && <FilesTab sessionId={sessionId} />}
        {activeTab === "artifacts" && <ArtifactsTab />}
      </div>
    );
  }

  function renderPreview() {
    if (!preview.activeArtifact || !preview.classification) return renderTabs();
    return (
      <>
        <ArtifactPanelHeader
          artifact={preview.activeArtifact}
          classification={preview.classification}
          canGoBack={preview.history.length > 0}
          isMaximized={preview.isMaximized}
          isSourceView={preview.isSourceView}
          hasSourceToggle={preview.classification.hasSourceToggle}
          mobile={!!mobile}
          canCopy={preview.canCopy}
          onBack={preview.goBackArtifact}
          onClose={preview.closeArtifactPanel}
          onMinimize={preview.minimizeArtifactPanel}
          onMaximize={preview.maximizeArtifactPanel}
          onRestore={preview.restoreArtifactPanel}
          onCopy={preview.handleCopy}
          onDownload={preview.handleDownload}
          onSourceToggle={preview.setIsSourceView}
        />
        <ArtifactContent
          artifact={preview.activeArtifact}
          isSourceView={preview.isSourceView}
          classification={preview.classification}
        />
      </>
    );
  }

  // The tabs (file list / progress) blend into the gray canvas, but an
  // artifact preview needs a readable white surface — so it pops as a card
  // like the chat column.
  const isPreviewing =
    view === "preview" && !!preview.activeArtifact && !!preview.classification;
  const body = view === "preview" ? renderPreview() : renderTabs();

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
            <SheetTitle>
              {preview.activeArtifact?.title ?? "Workspace"}
            </SheetTitle>
          </SheetHeader>
          {body}
        </SheetContent>
      </Sheet>
    );
  }

  if (isOpen && preview.isMinimized && view === "preview") {
    return (
      <ArtifactMinimizedStrip
        artifact={preview.activeArtifact!}
        classification={preview.classification!}
        onExpand={preview.restoreArtifactPanel}
      />
    );
  }

  return (
    <AnimatePresence initial={false}>
      {isOpen && (
        // data-artifact-panel is required by the reused ArtifactDragHandle,
        // which measures the panel via closest("[data-artifact-panel]").
        <motion.div
          key="context-panel"
          data-context-panel
          data-artifact-panel
          initial={{ width: 0 }}
          animate={{ width: "20rem" }}
          exit={{ width: 0 }}
          transition={{ duration: 0.2, ease: "linear" }}
          className={cn(
            "relative flex h-full shrink-0 flex-col overflow-hidden",
            isPreviewing
              ? "rounded-2xl border border-zinc-200 bg-white shadow-sm"
              : "bg-transparent",
          )}
        >
          {/* Inner stays at the target width so contents don't reflow during
              the open/close animation — the outer just reveals/hides it.
              Opacity is delayed so the content fades in after the width
              animation completes, avoiding the squished-content look. */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { duration: 0.1, delay: 0 } }}
            transition={{ duration: 0.15, delay: 0.2 }}
            className="flex h-full w-[20rem] flex-col"
          >
            <ArtifactDragHandle onWidthChange={setArtifactPanelWidth} />
            {body}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
