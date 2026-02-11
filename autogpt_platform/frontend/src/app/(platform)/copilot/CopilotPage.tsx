"use client";

import { SidebarProvider } from "@/components/ui/sidebar";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { ChatSidebar } from "./components/ChatSidebar/ChatSidebar";
import { MobileDrawer } from "./components/MobileDrawer/MobileDrawer";
import { MobileHeader } from "./components/MobileHeader/MobileHeader";
import { ScaleLoader } from "./components/ScaleLoader/ScaleLoader";
import { useCopilotPage } from "./useCopilotPage";

export function CopilotPage() {
  const {
    sessionId,
    messages,
    status,
    error,
    stop,
    createSession,
    onSend,
    isLoadingSession,
    isCreatingSession,
    isUserLoading,
    isLoggedIn,
    // Mobile drawer
    isMobile,
    isDrawerOpen,
    sessions,
    isLoadingSessions,
    handleOpenDrawer,
    handleCloseDrawer,
    handleDrawerOpenChange,
    handleSelectSession,
    handleNewChat,
  } = useCopilotPage();

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
      <div className="relative flex h-full w-full flex-col overflow-hidden bg-[#f8f8f9] px-0">
        {isMobile && <MobileHeader onOpenDrawer={handleOpenDrawer} />}
        <div className="flex-1 overflow-hidden">
          <ChatContainer
            messages={messages}
            status={status}
            error={error}
            sessionId={sessionId}
            isLoadingSession={isLoadingSession}
            isCreatingSession={isCreatingSession}
            onCreateSession={createSession}
            onSend={onSend}
            onStop={stop}
          />
        </div>
      </div>
      {isMobile && (
        <MobileDrawer
          isOpen={isDrawerOpen}
          sessions={sessions}
          currentSessionId={sessionId}
          isLoading={isLoadingSessions}
          onSelectSession={handleSelectSession}
          onNewChat={handleNewChat}
          onClose={handleCloseDrawer}
          onOpenChange={handleDrawerOpenChange}
        />
      )}
    </SidebarProvider>
  );
}
