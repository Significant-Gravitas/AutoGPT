"use client";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { SidebarProvider } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { DotsThree, UploadSimple } from "@phosphor-icons/react";
import { useCallback, useRef, useState } from "react";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { ALLOWED_EXTENSIONS } from "./components/ChatInput/components/AttachmentMenu";
import { ChatSidebar } from "./components/ChatSidebar/ChatSidebar";
import { DeleteChatDialog } from "./components/DeleteChatDialog/DeleteChatDialog";
import { MobileDrawer } from "./components/MobileDrawer/MobileDrawer";
import { MobileHeader } from "./components/MobileHeader/MobileHeader";
import { ScaleLoader } from "./components/ScaleLoader/ScaleLoader";
import { useCopilotPage } from "./useCopilotPage";

function getFileExtension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot).toLowerCase();
}

export function CopilotPage() {
  const { toast } = useToast();
  const [isDragging, setIsDragging] = useState(false);
  const [droppedFiles, setDroppedFiles] = useState<File[]>([]);
  const dragCounter = useRef(0);

  const handleDroppedFilesConsumed = useCallback(() => {
    setDroppedFiles([]);
  }, []);

  function handleDragEnter(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current += 1;
    if (e.dataTransfer.types.includes("Files")) {
      setIsDragging(true);
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current -= 1;
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current = 0;
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;

    const accepted: File[] = [];
    const rejected: string[] = [];

    for (const file of files) {
      const ext = getFileExtension(file.name);
      if (ext && ALLOWED_EXTENSIONS.has(ext)) {
        accepted.push(file);
      } else {
        rejected.push(file.name);
      }
    }

    if (rejected.length > 0) {
      toast({
        title: "Unsupported file type",
        description: `${rejected.join(", ")} — only documents, images, audio, video, and spreadsheets are supported.`,
        variant: "destructive",
      });
    }

    if (accepted.length > 0) {
      setDroppedFiles(accepted);
    }
  }

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
    isUploadingFiles,
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
      <div
        className="relative flex h-full w-full flex-col overflow-hidden bg-[#f8f8f9] px-0"
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isMobile && <MobileHeader onOpenDrawer={handleOpenDrawer} />}
        {/* Drop overlay */}
        <div
          className={cn(
            "pointer-events-none absolute inset-0 z-50 flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed border-violet-400 bg-violet-500/10 transition-opacity duration-150",
            isDragging ? "opacity-100" : "opacity-0",
          )}
        >
          <UploadSimple className="h-10 w-10 text-violet-500" weight="bold" />
          <span className="text-lg font-medium text-violet-600">
            Drop files here
          </span>
        </div>
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
            isUploadingFiles={isUploadingFiles}
            droppedFiles={droppedFiles}
            onDroppedFilesConsumed={handleDroppedFilesConsumed}
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
