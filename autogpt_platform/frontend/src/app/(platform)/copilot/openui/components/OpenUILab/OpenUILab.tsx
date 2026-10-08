"use client";

import { useOpenUILab } from "../../useOpenUILab";
import { ConversationPanel } from "../ConversationPanel/ConversationPanel";
import { LabHeader } from "../LabHeader";
import { LabSidebar } from "../LabSidebar";
import { ScenarioPicker } from "../ScenarioPicker";
import { WorkspacePanel } from "../WorkspacePanel/WorkspacePanel";
import { useState } from "react";
import { cn } from "@/lib/utils";

interface Props {
  standalone?: boolean;
  liveAvailable: boolean;
}

export function OpenUILab({ standalone = false, liveAvailable }: Props) {
  const lab = useOpenUILab(liveAvailable);
  const [mobileView, setMobileView] = useState("Workspace");
  return (
    <div
      className={cn(
        "flex min-h-[40rem] bg-white font-sans",
        standalone ? "h-dvh" : "h-[calc(100dvh-4rem)]",
      )}
    >
      {standalone && <LabSidebar />}
      <div className="flex min-w-0 flex-1 flex-col">
        <LabHeader />
        <ScenarioPicker selected={lab.scenario} onSelect={lab.selectScenario} />
        {lab.error && (
          <p
            role="alert"
            className="mx-5 mb-3 rounded-lg bg-red-50 px-4 py-3 text-xs leading-relaxed text-red-700 sm:mx-8"
          >
            {lab.error}
          </p>
        )}
        <div
          className="flex shrink-0 gap-2 px-5 pb-3 sm:px-8 lg:hidden"
          aria-label="Lab panels"
        >
          {["Workspace", "Conversation"].map((value) => (
            <button
              key={value}
              aria-pressed={mobileView === value}
              onClick={() => setMobileView(value)}
              className={cn(
                "rounded-lg px-3 py-2 text-xs font-medium text-zinc-500",
                mobileView === value && "bg-zinc-100 text-zinc-900",
              )}
            >
              {value}
            </button>
          ))}
        </div>
        <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 px-5 pb-5 sm:px-8 lg:grid-cols-[17rem_minmax(0,1fr)] xl:grid-cols-[18rem_minmax(0,1fr)]">
          <div
            className={cn(
              "min-h-0 flex-col lg:flex",
              mobileView !== "Conversation" && "hidden",
            )}
          >
            <ConversationPanel
              messages={lab.messages}
              prompt={lab.prompt}
              onPrompt={lab.setPrompt}
              mode={lab.mode}
              onMode={lab.changeMode}
              liveAvailable={liveAvailable}
              standalone={standalone}
              isStreaming={lab.isStreaming}
              suggestions={lab.scenario.suggestions}
              onSend={lab.send}
              onStop={lab.stop}
              onSubmit={lab.handleSubmit}
              onKeyDown={lab.handleKeyDown}
            />
          </div>
          <div
            className={cn(
              "min-h-0 min-w-0 flex-col lg:flex",
              mobileView !== "Workspace" && "hidden",
            )}
          >
            <WorkspacePanel
              source={lab.source}
              isStreaming={lab.isStreaming}
              revision={lab.revision}
              mode={lab.sourceMode}
              onAction={lab.handleAction}
              onReplay={lab.replay}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
