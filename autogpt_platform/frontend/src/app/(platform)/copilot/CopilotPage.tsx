"use client";

import { LowCreditBanner } from "@/components/layout/TopUpPrompt/LowCreditBanner/LowCreditBanner";
import { DotDistortionShader } from "@/components/ui/dot-distortion-shader";
import { SidebarProvider } from "@/components/ui/sidebar";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import dynamic from "next/dynamic";
import { parseAsString, useQueryState } from "nuqs";
import { useState } from "react";
import { CopilotChatHost } from "./CopilotChatHost";
import { ContextPanelAutoOpen } from "./components/ContextPanel/ContextPanelAutoOpen";
import { ContextPanelToggle } from "./components/ContextPanel/ContextPanelToggle";
import { ChatSidebar } from "./components/ChatSidebar/ChatSidebar";
import { FileDropZone } from "./components/FileDropZone/FileDropZone";
import { MobileDrawer } from "./components/MobileDrawer/MobileDrawer";
import { MobileHeader } from "./components/MobileHeader/MobileHeader";
import { NotificationBanner } from "./components/NotificationBanner/NotificationBanner";
import { NotificationDialog } from "./components/NotificationDialog/NotificationDialog";
import { ScaleLoader } from "./components/ScaleLoader/ScaleLoader";
import { useIsMobile } from "./useIsMobile";

const ArtifactPanel = dynamic(
  () =>
    import("./components/ArtifactPanel/ArtifactPanel").then(
      (m) => m.ArtifactPanel,
    ),
  { ssr: false },
);

const ContextPanel = dynamic(
  () =>
    import("./components/ContextPanel/ContextPanel").then(
      (m) => m.ContextPanel,
    ),
  { ssr: false },
);

export function CopilotPage() {
  const [droppedFiles, setDroppedFiles] = useState<File[]>([]);
  const isMobile = useIsMobile();
  const isArtifactsEnabled = useGetFlag(Flag.ARTIFACTS);
  const { isUserLoading, isLoggedIn } = useSupabase();
  // Read sessionId here purely to key the chat-host subtree. The view still
  // remounts on session switch, but the underlying AI SDK Chat runtime now
  // lives in a per-session registry so live streams can continue in
  // background JS state while another chat is on screen.
  const [sessionId] = useQueryState("sessionId", parseAsString);

  if (isUserLoading || !isLoggedIn) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#f8f8f9]">
        <ScaleLoader className="text-neutral-400" />
      </div>
    );
  }

  return (
    <SidebarProvider
      defaultOpen={true}
      // Explicit height: `h-full` against <section className="flex-1"> drifts
      // out of sync with the navbar-driven --preview-banner-height var during
      // re-renders, clipping the navbar when the sidebar toggles.
      style={{
        height: `calc(100vh - ${NAVBAR_HEIGHT_PX}px - var(--preview-banner-height, 0px))`,
      }}
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
      {isMobile && isArtifactsEnabled && sessionId && (
        <ContextPanel sessionId={sessionId} mobile />
      )}
      {isMobile && isArtifactsEnabled && <ArtifactPanel mobile />}
      {isMobile && <MobileDrawer />}
      <NotificationDialog />
    </SidebarProvider>
  );
}

interface MainAreaProps {
  isMobile: boolean;
  isArtifactsEnabled: boolean;
  sessionId: string | null;
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
  const hasSession = !!sessionId;
  return (
    <div className="flex h-full w-full flex-row overflow-hidden">
      <div className="relative flex min-w-0 flex-1 overflow-hidden bg-[#fafafa]">
        {hasSession && (
          <DotDistortionShader
            dotGap={14}
            dotSize={1}
            opacity={0.2}
            isStatic
            className="pointer-events-none absolute inset-0 !bg-transparent [&_canvas]:opacity-70"
          />
        )}
        <FileDropZone
          className="relative flex min-w-0 flex-1 flex-col overflow-hidden px-0"
          onFilesDropped={setDroppedFiles}
        >
          {isMobile && <MobileHeader />}
          <div className="flex flex-col gap-3 px-4 pt-4 empty:hidden">
            <LowCreditBanner />
            <NotificationBanner />
          </div>
          <CopilotChatHost
            key={`chat-host-${sessionId ?? "new"}`}
            droppedFiles={droppedFiles}
            onDroppedFilesConsumed={() => setDroppedFiles([])}
          />
          {!isMobile && isArtifactsEnabled && (
            <ContextPanelAutoOpen
              key={`context-auto-open-${sessionId ?? "new"}`}
              sessionId={sessionId}
            />
          )}
        </FileDropZone>
      </div>
      {!isMobile && isArtifactsEnabled && sessionId && (
        <ContextPanel sessionId={sessionId} />
      )}
      {!isMobile && isArtifactsEnabled && sessionId && <ArtifactPanel />}
      {!isMobile && isArtifactsEnabled && sessionId && <ContextPanelToggle />}
    </div>
  );
}
