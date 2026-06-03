"use client";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { AnimatePresence, motion } from "framer-motion";
import { ArtifactContent } from "./components/ArtifactContent";
import { ArtifactDragHandle } from "./components/ArtifactDragHandle";
import { ArtifactMinimizedStrip } from "./components/ArtifactMinimizedStrip";
import { ArtifactPanelHeader } from "./components/ArtifactPanelHeader";
import { useArtifactPanel } from "./useArtifactPanel";

interface Props {
  mobile?: boolean;
}

export function ArtifactPanel({ mobile }: Props) {
  const {
    isOpen,
    isMinimized,
    isMaximized,
    activeArtifact,
    history,
    effectiveWidth,
    isSourceView,
    classification,
    setIsSourceView,
    closeArtifactPanel,
    minimizeArtifactPanel,
    maximizeArtifactPanel,
    restoreArtifactPanel,
    setArtifactPanelWidth,
    goBackArtifact,
    canCopy,
    handleCopy,
    handleDownload,
  } = useArtifactPanel();

  if (!activeArtifact || !classification) return null;

  const headerProps = {
    artifact: activeArtifact,
    classification,
    canGoBack: history.length > 0,
    isMaximized,
    isSourceView,
    hasSourceToggle: classification.hasSourceToggle,
    mobile: !!mobile,
    canCopy,
    onBack: goBackArtifact,
    onClose: closeArtifactPanel,
    onMinimize: minimizeArtifactPanel,
    onMaximize: maximizeArtifactPanel,
    onRestore: restoreArtifactPanel,
    onCopy: handleCopy,
    onDownload: handleDownload,
    onSourceToggle: setIsSourceView,
  };

  // Mobile: fullscreen Sheet overlay
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
            <SheetTitle>{activeArtifact.title}</SheetTitle>
          </SheetHeader>
          <ArtifactPanelHeader {...headerProps} />
          <ArtifactContent
            artifact={activeArtifact}
            isSourceView={isSourceView}
            classification={classification}
          />
        </SheetContent>
      </Sheet>
    );
  }

  // Minimized strip
  if (isOpen && isMinimized) {
    return (
      <ArtifactMinimizedStrip
        artifact={activeArtifact}
        classification={classification}
        onExpand={restoreArtifactPanel}
      />
    );
  }

  // Keep AnimatePresence mounted across the open→closed transition so the
  // exit animation on the motion.div has a chance to run.
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          key="artifact-panel"
          data-artifact-panel
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.25, ease: "easeInOut" }}
          className="relative flex h-full flex-col overflow-hidden border-l border-zinc-200 bg-white"
          style={{ width: effectiveWidth }}
        >
          <ArtifactDragHandle onWidthChange={setArtifactPanelWidth} />
          <ArtifactPanelHeader {...headerProps} />
          <ArtifactContent
            artifact={activeArtifact}
            isSourceView={isSourceView}
            classification={classification}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
