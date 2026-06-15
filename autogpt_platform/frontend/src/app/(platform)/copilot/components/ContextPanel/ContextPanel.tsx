"use client";

import { Button } from "@/components/ui/button";
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
import { ContextPanelRail } from "./components/ContextPanelRail";
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
    showRail,
    showExpanded,
    setActiveTab,
    closeArtifactPanel,
    expandContextPanel,
    contextPanelWidth,
    setContextPanelWidth,
  } = useContextPanel();
  const { uploaded, generated } = useSessionFiles(sessionId);
  const filesCount = uploaded.length + generated.length;

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
      <div className={cn("p-2", mobile && "mt-12")}>
        <TabSwitcher
          activeTab={activeTab}
          filesCount={filesCount}
          onChange={setActiveTab}
        />
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

  if (showRail) {
    return <ContextPanelRail onExpand={expandContextPanel} />;
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
