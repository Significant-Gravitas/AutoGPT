"use client";

import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { XIcon } from "@phosphor-icons/react";
import { AnimatePresence, motion } from "framer-motion";
import { ArtifactDragHandle } from "../ArtifactPanel/components/ArtifactDragHandle";
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
    width,
    setActiveTab,
    setArtifactPanelWidth,
    closeArtifactPanel,
  } = useContextPanel();

  const tabs = (
    <div className="flex min-h-0 flex-1 flex-col">
      {!mobile && (
        <div className="flex justify-end p-2">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={closeArtifactPanel}
            aria-label="Close workspace panel"
          >
            <XIcon className="!size-5" />
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
          animate={{ width }}
          exit={{ width: 0 }}
          transition={{ duration: 0.2, ease: "linear" }}
          className="relative flex h-full shrink-0 flex-col overflow-hidden bg-transparent"
        >
          {/* Opacity is delayed so the content fades in after the width
              animation completes, avoiding the squished-content look. */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { duration: 0.1, delay: 0 } }}
            transition={{ duration: 0.15, delay: 0.2 }}
            className="flex h-full w-full flex-col"
          >
            <ArtifactDragHandle
              onWidthChange={setArtifactPanelWidth}
              minWidth={240}
              maxWidth={280}
            />
            {tabs}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
