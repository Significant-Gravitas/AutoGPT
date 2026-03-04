"use client";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { SidebarProvider } from "@/components/ui/sidebar";
import { DotsThree } from "@phosphor-icons/react";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { ChatSidebar } from "./components/ChatSidebar/ChatSidebar";
import { DeleteChatDialog } from "./components/DeleteChatDialog/DeleteChatDialog";
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
    isReconnecting,
    createSession,
    onSend,
    isLoadingSession,
    isSessionError,
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
    // Delete functionality
    sessionToDelete,
    isDeleting,
    handleDeleteClick,
    handleConfirmDelete,
    handleCancelDelete,
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
            isSessionError={isSessionError}
            isCreatingSession={isCreatingSession}
            isReconnecting={isReconnecting}
            onCreateSession={createSession}
            onSend={onSend}
            onStop={stop}
            headerSlot={
              isMobile && sessionId ? (
                <div className="flex justify-end">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button
                        className="rounded p-1.5 hover:bg-neutral-100"
                        aria-label="More actions"
                      >
                        <DotsThree className="h-5 w-5 text-neutral-600" />
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onClick={() => {
                          const session = sessions.find(
                            (s) => s.id === sessionId,
                          );
                          if (session) {
                            handleDeleteClick(session.id, session.title);
                          }
                        }}
                        disabled={isDeleting}
                        className="text-red-600 focus:bg-red-50 focus:text-red-600"
                      >
                        Delete chat
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              ) : undefined
            }
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
      {/* Delete confirmation dialog - rendered at top level for proper z-index on mobile */}
      {isMobile && (
        <DeleteChatDialog
          session={sessionToDelete}
          isDeleting={isDeleting}
          onConfirm={handleConfirmDelete}
          onCancel={handleCancelDelete}
        />
      )}
    </SidebarProvider>
  );
}
