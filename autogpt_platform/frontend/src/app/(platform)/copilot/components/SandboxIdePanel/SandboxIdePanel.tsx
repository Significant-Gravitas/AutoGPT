"use client";

import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { XIcon } from "@phosphor-icons/react";
import { MIN_SANDBOX_IDE_PANEL_WIDTH, type SandboxIdeTab } from "../../store";
import { PanelResizeHandle } from "../PanelResizeHandle";
import { ChangesCountBadge } from "./components/ChangesTab/ChangesCountBadge";
import { ChangesTab } from "./components/ChangesTab/ChangesTab";
import { DownloadButton } from "./components/DownloadButton/DownloadButton";
import { FilesTab } from "./components/FilesTab/FilesTab";
import { TerminalTab } from "./components/TerminalTab/TerminalTab";
import { useSandboxIdePanel } from "./useSandboxIdePanel";

interface Props {
  sessionId: string;
}

export function SandboxIdePanel({ sessionId }: Props) {
  const { isOpen, activeTab, width, setActiveTab, close, setWidth } =
    useSandboxIdePanel();

  if (!isOpen) return null;

  return (
    <div
      data-sandbox-ide-panel
      style={{ width }}
      className="relative flex h-full shrink-0 flex-col border-l border-l-[#80808017] bg-sidebar"
    >
      <PanelResizeHandle
        panelSelector="[data-sandbox-ide-panel]"
        onWidthChange={setWidth}
        minWidth={MIN_SANDBOX_IDE_PANEL_WIDTH}
      />
      <div className="flex items-center justify-between gap-2 border-b border-b-[#80808017] px-3 py-2">
        <span className="text-sm font-medium text-zinc-700">Sandbox</span>
        <div className="flex items-center gap-1">
          <DownloadButton sessionId={sessionId} />
          <button
            type="button"
            aria-label="Close sandbox panel"
            onClick={close}
            className="rounded p-1 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-800"
          >
            <XIcon size={16} />
          </button>
        </div>
      </div>
      <TabsLine
        value={activeTab}
        onValueChange={(value) => setActiveTab(value as SandboxIdeTab)}
        className="flex min-h-0 flex-1 flex-col"
      >
        <TabsLineList flush className="px-3">
          <TabsLineTrigger value="files">Files</TabsLineTrigger>
          <TabsLineTrigger value="changes">
            Changes
            <ChangesCountBadge sessionId={sessionId} />
          </TabsLineTrigger>
          <TabsLineTrigger value="terminal">Terminal</TabsLineTrigger>
        </TabsLineList>
        <TabsLineContent
          value="files"
          className="mt-0 min-h-0 flex-1 overflow-hidden"
        >
          <FilesTab sessionId={sessionId} />
        </TabsLineContent>
        <TabsLineContent
          value="changes"
          className="mt-0 min-h-0 flex-1 overflow-hidden"
        >
          <ChangesTab sessionId={sessionId} />
        </TabsLineContent>
        <TabsLineContent
          value="terminal"
          className="mt-0 min-h-0 flex-1 overflow-hidden"
        >
          <TerminalTab sessionId={sessionId} />
        </TabsLineContent>
      </TabsLine>
    </div>
  );
}
