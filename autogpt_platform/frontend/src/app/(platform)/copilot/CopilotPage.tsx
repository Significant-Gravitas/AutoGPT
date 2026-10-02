"use client";

import { LowCreditBanner } from "@/components/layout/TopUpPrompt/LowCreditBanner/LowCreditBanner";
import { SidebarProvider } from "@/components/ui/sidebar";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import dynamic from "next/dynamic";
import { parseAsString, useQueryState } from "nuqs";
import { useState } from "react";
import { CopilotChatHost } from "./CopilotChatHost";
import { ContextPanelAutoOpen } from "./components/ContextPanel/ContextPanelAutoOpen";
import { CopilotModals } from "./components/CopilotModals/CopilotModals";
import { FileDropZone } from "./components/FileDropZone/FileDropZone";
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
  const isBrainDumpEnabled = useGetFlag(Flag.ONBOARDING_BRAIN_DUMP);
  const { isUserLoading, isLoggedIn } = useAuth();
  // Read sessionId here purely to key the chat-host subtree. The view still
  // remounts on session switch, but the underlying AI SDK Chat runtime now
  // lives in a per-session registry so live streams can continue in
  // background JS state while another chat is on screen.
  const [sessionId] = useQueryState("sessionId", parseAsString);

  if (isUserLoading || !isLoggedIn) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#f6f7fb]">
        <ScaleLoader className="text-neutral-400" />
      </div>
    );
  }

  return (
    <SidebarProvider
      defaultOpen={true}
      // The chat column needs an explicit, viewport-bound height: it relies on
      // a definite height so its inner `min-h-0` chain lets the messages area
      // (not the page) absorb growth — e.g. expanding the task progress
      // accordion above the input. The chrome's ancestors (SidebarProvider
      // `min-h-svh` → SidebarInset `flex-1` → `section flex-1`) only set a
      // *minimum* height, so `height: 100%` would resolve to content height
      // and the accordion would push the input below the fold. The inset
      // header overlays the chat (see PlatformChrome) instead of stacking
      // above it, so the full viewport is available. `svh` keeps the input
      // visible when mobile browser chrome is shown.
      style={{ height: "100svh" }}
      className="min-h-0"
    >
      <MainArea
        isMobile={isMobile}
        sessionId={sessionId}
        droppedFiles={droppedFiles}
        setDroppedFiles={setDroppedFiles}
      />
      {isMobile && sessionId && <ContextPanel sessionId={sessionId} mobile />}
      {isMobile && <ArtifactPanel mobile />}
      {!isBrainDumpEnabled && <NotificationDialog />}
      <CopilotModals />
    </SidebarProvider>
  );
}

interface MainAreaProps {
  isMobile: boolean;
  sessionId: string | null;
  droppedFiles: File[];
  setDroppedFiles: (files: File[]) => void;
}

function MainArea({
  isMobile,
  sessionId,
  droppedFiles,
  setDroppedFiles,
}: MainAreaProps) {
  return (
    <div className="flex h-full w-full flex-row overflow-hidden">
      <div className="relative flex min-w-0 flex-1 overflow-hidden bg-[#f6f7fb]">
        <FileDropZone
          className="relative flex min-w-0 flex-1 flex-col overflow-hidden px-0"
          onFilesDropped={setDroppedFiles}
        >
          {/* max-lg:pt-16 clears the floating inset-header controls (sidebar
              toggle + workspace-files trigger) that overlay the top-left
              corner. */}
          <div className="flex flex-col gap-3 px-4 pt-4 empty:hidden max-lg:pt-16">
            <LowCreditBanner />
            <NotificationBanner />
          </div>
          <CopilotChatHost
            key={`chat-host-${sessionId ?? "new"}`}
            droppedFiles={droppedFiles}
            onDroppedFilesConsumed={() => setDroppedFiles([])}
            hasFloatingControls
          />
          {/* Owns the session-entry reset that forgets the previous chat's
              artifact. */}
          <ContextPanelAutoOpen
            key={`context-auto-open-${sessionId ?? "new"}`}
            sessionId={sessionId}
          />
        </FileDropZone>
      </div>
      {!isMobile && sessionId && <ContextPanel sessionId={sessionId} />}
      {!isMobile && sessionId && <ArtifactPanel sessionId={sessionId} />}
    </div>
  );
}
