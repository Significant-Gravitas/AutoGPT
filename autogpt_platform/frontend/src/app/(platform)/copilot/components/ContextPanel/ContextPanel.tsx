"use client";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import { XIcon } from "@phosphor-icons/react";
import { MAX_CONTEXT_PANEL_WIDTH, MIN_CONTEXT_PANEL_WIDTH } from "../../store";
import { PanelResizeHandle } from "../PanelResizeHandle";
import { FilesTab } from "./components/FilesTab/FilesTab";
import { useSessionFiles } from "./components/FilesTab/useSessionFiles";
import { ProgressTab } from "./components/ProgressTab/ProgressTab";
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
    showExpanded,
    setActiveTab,
    closeArtifactPanel,
    contextPanelWidth,
    setContextPanelWidth,
  } = useContextPanel();
  const { uploaded, generated } = useSessionFiles(sessionId);
  const filesCount = uploaded.length + generated.length;

  const tabs = (
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        className={cn(
          "flex items-center justify-between gap-2 p-2",
          mobile && "mt-12",
        )}
      >
        <TabSwitcher
          activeTab={activeTab}
          filesCount={filesCount}
          onChange={setActiveTab}
        />
        {!mobile && (
          <button
            type="button"
            onClick={closeArtifactPanel}
            title="Close"
            aria-label="Close workspace panel"
            className="rounded p-1.5 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
          >
            <XIcon size={16} />
          </button>
        )}
      </div>
      <div className="flex min-h-0 flex-1 flex-col">
        {activeTab === "progress" && <ProgressTab sessionId={sessionId} />}
        {activeTab === "files" && <FilesTab sessionId={sessionId} />}
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

  if (!showExpanded) return null;

  return (
    <div
      data-context-panel
      style={{ width: contextPanelWidth }}
      className="relative flex h-full shrink-0 flex-col border-l border-l-[#80808017] bg-sidebar"
    >
      <PanelResizeHandle
        panelSelector="[data-context-panel]"
        onWidthChange={setContextPanelWidth}
        minWidth={MIN_CONTEXT_PANEL_WIDTH}
        maxWidth={MAX_CONTEXT_PANEL_WIDTH}
      />
      <div className="flex min-h-0 w-full flex-1 flex-col overflow-hidden">
        {tabs}
      </div>
    </div>
  );
}
