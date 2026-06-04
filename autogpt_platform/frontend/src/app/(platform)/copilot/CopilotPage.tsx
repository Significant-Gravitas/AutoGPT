"use client";

import { LowCreditBanner } from "@/components/layout/TopUpPrompt/LowCreditBanner/LowCreditBanner";
import { SidebarProvider } from "@/components/ui/sidebar";
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
  const isContextPanelEnabled = useGetFlag(Flag.CONTEXT_PANEL);
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
      className="h-[calc(100vh-72px)] min-h-0"
    >
      {!isMobile && <ChatSidebar />}
      <MainArea
        isMobile={isMobile}
        isContextPanelEnabled={isContextPanelEnabled}
        sessionId={sessionId}
        droppedFiles={droppedFiles}
        setDroppedFiles={setDroppedFiles}
      />
      {isMobile && isContextPanelEnabled && (
        <ContextPanel sessionId={sessionId} mobile />
      )}
      {isArtifactsEnabled && <ArtifactPanel mobile={isMobile} />}
      {isMobile && <MobileDrawer />}
      <NotificationDialog />
    </SidebarProvider>
  );
}

interface MainAreaProps {
  isMobile: boolean;
  isContextPanelEnabled: boolean;
  sessionId: string | null;
  droppedFiles: File[];
  setDroppedFiles: (files: File[]) => void;
}

function MainArea({
  isMobile,
  isContextPanelEnabled,
  sessionId,
  droppedFiles,
  setDroppedFiles,
}: MainAreaProps) {
  return (
    <div className="relative mr-5 mt-2.5 flex h-full w-full flex-row pb-1 lg:mr-[0.3rem]">
      <div className="relative flex min-w-0 flex-1 overflow-hidden bg-[#fafafa] p-2">
        <FileDropZone
          className="relative flex min-w-0 flex-1 flex-col overflow-hidden rounded-2xl border border-[#8080801a] bg-white px-0 shadow-sm"
          onFilesDropped={setDroppedFiles}
        >
          {isMobile && <MobileHeader />}
          <LowCreditBanner className="m-4" />
          <NotificationBanner />
          <CopilotChatHost
            key={`chat-host-${sessionId ?? "new"}`}
            droppedFiles={droppedFiles}
            onDroppedFilesConsumed={() => setDroppedFiles([])}
          />
          {/* Auto-open is desktop-only: on mobile the panel is a fullscreen
              sheet, so opening it on first file would take over the chat. */}
          {!isMobile && isContextPanelEnabled && (
            <ContextPanelAutoOpen
              key={`context-auto-open-${sessionId ?? "new"}`}
              sessionId={sessionId}
            />
          )}
        </FileDropZone>
      </div>
      {!isMobile && isContextPanelEnabled && (
        <ContextPanel sessionId={sessionId} />
      )}
      {!isMobile && isContextPanelEnabled && sessionId && <ContextPanelToggle />}
    </div>
  );
}
