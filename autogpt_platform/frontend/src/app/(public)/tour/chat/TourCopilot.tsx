"use client";

import { ContextPanelAutoOpen } from "@/app/(platform)/copilot/components/ContextPanel/ContextPanelAutoOpen";
import { ContextPanelToggle } from "@/app/(platform)/copilot/components/ContextPanel/ContextPanelToggle";
import { ChatSidebar } from "@/app/(platform)/copilot/components/ChatSidebar/ChatSidebar";
import { FileDropZone } from "@/app/(platform)/copilot/components/FileDropZone/FileDropZone";
import { MobileHeader } from "@/app/(platform)/copilot/components/MobileHeader/MobileHeader";
import { useIsMobile } from "@/app/(platform)/copilot/useIsMobile";
import { DotDistortionShader } from "@/components/ui/dot-distortion-shader";
import { SidebarProvider } from "@/components/ui/sidebar";
import dynamic from "next/dynamic";
import { useState } from "react";
import { TourChatHost } from "./TourChatHost";
import { TOUR_SESSION_ID } from "./script/mockSidebarSessions";

const ArtifactPanel = dynamic(
  () =>
    import(
      "@/app/(platform)/copilot/components/ArtifactPanel/ArtifactPanel"
    ).then((m) => m.ArtifactPanel),
  { ssr: false },
);

const ContextPanel = dynamic(
  () =>
    import(
      "@/app/(platform)/copilot/components/ContextPanel/ContextPanel"
    ).then((m) => m.ContextPanel),
  { ssr: false },
);

export function TourCopilot() {
  const [droppedFiles, setDroppedFiles] = useState<File[]>([]);
  const isMobile = useIsMobile();
  const isArtifactsEnabled = true;
  const sessionId = TOUR_SESSION_ID;

  return (
    <SidebarProvider
      defaultOpen={true}
      style={{ height: "100dvh" }}
      className="min-h-0"
    >
      {!isMobile && <ChatSidebar />}
      <MainArea
        isMobile={isMobile}
        isArtifactsEnabled={isArtifactsEnabled}
        sessionId={sessionId}
        droppedFiles={droppedFiles}
        setDroppedFiles={setDroppedFiles}
      />
      {isMobile && isArtifactsEnabled && (
        <ContextPanel sessionId={sessionId} mobile />
      )}
      {isMobile && isArtifactsEnabled && <ArtifactPanel mobile />}
    </SidebarProvider>
  );
}

interface MainAreaProps {
  isMobile: boolean;
  isArtifactsEnabled: boolean;
  sessionId: string;
  droppedFiles: File[];
  setDroppedFiles: (files: File[]) => void;
}

function MainArea({
  isMobile,
  isArtifactsEnabled,
  sessionId,
  droppedFiles,
  setDroppedFiles,
}: MainAreaProps) {
  return (
    <div className="flex h-full w-full flex-row overflow-hidden">
      <div className="relative flex min-w-0 flex-1 overflow-hidden bg-[#fafafa]">
        <DotDistortionShader
          dotGap={14}
          dotSize={1}
          opacity={0.2}
          isStatic
          className="pointer-events-none absolute inset-0 !bg-transparent [&_canvas]:opacity-70"
        />
        <FileDropZone
          className="relative flex min-w-0 flex-1 flex-col overflow-hidden px-0"
          onFilesDropped={setDroppedFiles}
        >
          {isMobile && <MobileHeader />}
          <TourChatHost
            droppedFiles={droppedFiles}
            onDroppedFilesConsumed={() => setDroppedFiles([])}
          />
          {!isMobile && isArtifactsEnabled && (
            <ContextPanelAutoOpen sessionId={sessionId} />
          )}
        </FileDropZone>
      </div>
      {!isMobile && isArtifactsEnabled && (
        <ContextPanel sessionId={sessionId} />
      )}
      {!isMobile && isArtifactsEnabled && <ArtifactPanel />}
      {!isMobile && isArtifactsEnabled && <ContextPanelToggle />}
    </div>
  );
}
