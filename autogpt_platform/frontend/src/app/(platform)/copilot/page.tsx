"use client";

import { SidebarProvider } from "@/components/ui/sidebar";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { ChatSidebar } from "./components/ChatSidebar/ChatSidebar";
import { MobileDrawer } from "./components/MobileDrawer/MobileDrawer";
import { MobileHeader } from "./components/MobileHeader/MobileHeader";
import { useCopilotPage } from "./useCopilotPage";

export default function Page() {
  const {
    sessionId,
    messages,
    status,
    error,
    stop,
    isLoadingSession,
    isCreatingSession,
    createSession,
    onSend,
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
