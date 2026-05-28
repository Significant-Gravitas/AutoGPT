"use client";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
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
        <div className="border-b border-zinc-200 p-2">
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
    <AnimatePresence>
      {isOpen && (
        // data-artifact-panel is required by the reused ArtifactDragHandle,
        // which measures the panel via closest("[data-artifact-panel]").
        <motion.div
          key="context-panel"
          data-context-panel
          data-artifact-panel
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2, ease: "easeInOut" }}
          className={cn(
            "relative flex h-full flex-col overflow-hidden",
            isPreviewing
              ? "rounded-2xl border border-zinc-200 bg-white shadow-sm"
              : "bg-transparent",
          )}
          style={{ width: preview.effectiveWidth }}
        >
          <ArtifactDragHandle onWidthChange={setArtifactPanelWidth} />
          {body}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
