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

  if (!activeArtifact || (!isOpen && !mobile)) return null;

  const headerProps = {
    artifact: activeArtifact,
    canGoBack: history.length > 0,
    isMaximized,
    isSourceView,
    hasSourceToggle: classification?.hasSourceToggle ?? false,
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
          />
        </SheetContent>
      </Sheet>
    );
  }

  if (!isOpen) return null;

  // Minimized strip
  if (isMinimized) {
    return (
      <ArtifactMinimizedStrip
        artifact={activeArtifact}
        onExpand={restoreArtifactPanel}
      />
    );
  }

  return (
    <AnimatePresence>
      <motion.div
        data-artifact-panel
        initial={{ width: 0, opacity: 0 }}
        animate={{ width: effectiveWidth, opacity: 1 }}
        exit={{ width: 0, opacity: 0 }}
        transition={{ duration: 0.25, ease: "easeInOut" }}
        className="relative flex h-full flex-col border-l border-zinc-200 bg-white"
        style={{ minWidth: 320 }}
      >
        <ArtifactDragHandle onWidthChange={setArtifactPanelWidth} />
        <ArtifactPanelHeader {...headerProps} />
        <ArtifactContent
          artifact={activeArtifact}
          isSourceView={isSourceView}
        />
      </motion.div>
    </AnimatePresence>
  );
}
