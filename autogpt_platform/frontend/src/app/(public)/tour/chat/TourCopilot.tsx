"use client";

import { FileDropZone } from "@/app/(platform)/copilot/components/FileDropZone/FileDropZone";
import { MobileHeader } from "@/app/(platform)/copilot/components/MobileHeader/MobileHeader";
import { useIsMobile } from "@/app/(platform)/copilot/useIsMobile";
import { DotDistortionShader } from "@/components/ui/dot-distortion-shader";
import { SidebarProvider } from "@/components/ui/sidebar";
import { TourChatHost } from "./TourChatHost";
import { TourChatSidebar } from "./components/TourChatSidebar/TourChatSidebar";
import { getTourChat } from "./script/tourChats";
import type { TourScript } from "./script/types";
import { useTourStore } from "./tourStore";

export function TourCopilot() {
  const isMobile = useIsMobile();
  const activeSessionId = useTourStore((s) => s.activeSessionId);
  const chat = getTourChat(activeSessionId);

  return (
    <SidebarProvider
      defaultOpen={true}
      style={{ height: "100dvh" }}
      className="min-h-0"
    >
      {!isMobile && <TourChatSidebar />}
      <MainArea isMobile={isMobile} sessionId={chat.id} script={chat.script} />
    </SidebarProvider>
  );
}

interface MainAreaProps {
  isMobile: boolean;
  sessionId: string;
  script: TourScript;
}

function MainArea({ isMobile, sessionId, script }: MainAreaProps) {
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
          onFilesDropped={() => {}}
        >
          {isMobile && <MobileHeader />}
          <TourChatHost key={sessionId} sessionId={sessionId} script={script} />
        </FileDropZone>
      </div>
    </div>
  );
}
