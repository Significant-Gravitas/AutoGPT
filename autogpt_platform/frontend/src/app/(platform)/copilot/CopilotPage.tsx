"use client";

import { SidebarProvider } from "@/components/ui/sidebar";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import dynamic from "next/dynamic";
import { parseAsString, useQueryState } from "nuqs";
import { useState } from "react";
import { CopilotChatHost } from "./CopilotChatHost";
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

export function CopilotPage() {
  const [droppedFiles, setDroppedFiles] = useState<File[]>([]);
  const isMobile = useIsMobile();
  const isArtifactsEnabled = useGetFlag(Flag.ARTIFACTS);
  const { isUserLoading, isLoggedIn } = useSupabase();
  // Read sessionId here purely to key the chat-host subtree — the inner
  // host's useChatSession reads the same URL param via nuqs so both stay
  // in sync. Keying the subtree guarantees that every session switch
  // mounts a fresh useChat Chat instance, eliminating orphaned AI-SDK
  // state from the previous session (which was causing stuck UI on
  // session-switch-back — full reload worked, switch-back didn't).
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
      <div className="flex h-full w-full flex-row overflow-hidden">
        <FileDropZone
          className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-[#f8f8f9] px-0"
          onFilesDropped={setDroppedFiles}
        >
          {isMobile && <MobileHeader />}
          <NotificationBanner />
          <CopilotChatHost
            key={sessionId ?? "new"}
            droppedFiles={droppedFiles}
            onDroppedFilesConsumed={() => setDroppedFiles([])}
          />
        </FileDropZone>
        {!isMobile && isArtifactsEnabled && <ArtifactPanel />}
      </div>
      {isMobile && isArtifactsEnabled && <ArtifactPanel mobile />}
      {isMobile && <MobileDrawer />}
      <NotificationDialog />
    </SidebarProvider>
  );
}
