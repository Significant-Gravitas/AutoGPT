"use client";

import type { CoPilotUsageStatus } from "@/app/api/__generated__/models/coPilotUsageStatus";
import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import useCredits from "@/hooks/useCredits";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { SidebarProvider } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { Flask, UploadSimple } from "@phosphor-icons/react";
import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { ChatSidebar } from "./components/ChatSidebar/ChatSidebar";
import { DeleteChatDialog } from "./components/DeleteChatDialog/DeleteChatDialog";
import { MobileDrawer } from "./components/MobileDrawer/MobileDrawer";
import { MobileHeader } from "./components/MobileHeader/MobileHeader";
import { NotificationBanner } from "./components/NotificationBanner/NotificationBanner";
import { NotificationDialog } from "./components/NotificationDialog/NotificationDialog";
import { RateLimitResetDialog } from "./components/RateLimitResetDialog/RateLimitResetDialog";
import { ScaleLoader } from "./components/ScaleLoader/ScaleLoader";
import { useCopilotPage } from "./useCopilotPage";

const ArtifactPanel = dynamic(
  () =>
    import("./components/ArtifactPanel/ArtifactPanel").then(
      (m) => m.ArtifactPanel,
    ),
  { ssr: false },
);

export function CopilotPage() {
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
    if (files.length > 0) {
      setDroppedFiles(files);
    }
  }

  const {
    sessionId,
    messages,
    status,
    error,
    stop,
    isReconnecting,
    isSyncing,
    createSession,
    onSend,
    onEnqueue,
    queuedMessages,
    isLoadingSession,
    isSessionError,
    isCreatingSession,
    isUploadingFiles,
    isUserLoading,
    isLoggedIn,
    // Pagination
    hasMoreMessages,
    isLoadingMore,
    loadMore,
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
    // Delete functionality (available via ChatSidebar context menu on all viewports)
    sessionToDelete,
    isDeleting,
    handleConfirmDelete,
    handleCancelDelete,
    // Historical durations for persisted timer stats
    historicalDurations,
    // Rate limit reset
    rateLimitMessage,
    dismissRateLimit,
    // Dry run session state
    sessionDryRun,
  } = useCopilotPage();

  const {
    data: usage,
    isSuccess: hasUsage,
    isError: usageError,
  } = useGetV2GetCopilotUsage({
    query: {
      select: (res) => res.data as CoPilotUsageStatus,
      refetchInterval: 30000,
      staleTime: 10000,
    },
  });
  const resetCost = usage?.reset_cost;

  const isBillingEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const isArtifactsEnabled = useGetFlag(Flag.ARTIFACTS);
  const { credits, fetchCredits } = useCredits({ fetchInitialCredits: true });
  const hasInsufficientCredits =
    credits !== null && resetCost != null && credits < resetCost;

  // Fall back to a toast when the credit-based reset feature is disabled or
  // when the usage query fails (so the user still gets feedback).
  useEffect(() => {
    if (
      rateLimitMessage &&
      (usageError || (hasUsage && (resetCost ?? 0) <= 0))
    ) {
      toast({
        title: "Usage limit reached",
        description: rateLimitMessage,
        variant: "destructive",
      });
      dismissRateLimit();
    }
  }, [rateLimitMessage, resetCost, hasUsage, usageError, dismissRateLimit]);

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
        <div
          className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-[#f8f8f9] px-0"
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {isMobile && <MobileHeader onOpenDrawer={handleOpenDrawer} />}
          <NotificationBanner />
          {/* Test mode banner: only shown when the CURRENT session is confirmed to be
              a dry_run session via its immutable metadata. Never shown based on the
              global isDryRun store preference alone — that only predicts future sessions
              and would mislead users browsing non-dry-run sessions while the toggle is on.
              The DryRunToggleButton (visible on new chats) already communicates the preference. */}
          {sessionId && sessionDryRun && (
            <div className="flex items-center justify-center gap-1.5 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-800">
              <Flask size={13} weight="bold" />
              Test mode — this session runs agents as simulation
            </div>
          )}
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
              isSyncing={isSyncing}
              onCreateSession={createSession}
              onSend={onSend}
              onStop={stop}
              onEnqueue={onEnqueue}
              queuedMessages={queuedMessages}
              isUploadingFiles={isUploadingFiles}
              hasMoreMessages={hasMoreMessages}
              isLoadingMore={isLoadingMore}
              onLoadMore={loadMore}
              droppedFiles={droppedFiles}
              onDroppedFilesConsumed={handleDroppedFilesConsumed}
              historicalDurations={historicalDurations}
            />
          </div>
        </div>
        {!isMobile && isArtifactsEnabled && <ArtifactPanel />}
      </div>
      {isMobile && isArtifactsEnabled && <ArtifactPanel mobile />}
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
      <NotificationDialog />
      <RateLimitResetDialog
        isOpen={!!rateLimitMessage && hasUsage && (resetCost ?? 0) > 0}
        onClose={dismissRateLimit}
        resetCost={resetCost ?? 0}
        resetMessage={rateLimitMessage ?? ""}
        isWeeklyExhausted={
          hasUsage &&
          usage.weekly.limit > 0 &&
          usage.weekly.used >= usage.weekly.limit
        }
        hasInsufficientCredits={hasInsufficientCredits}
        isBillingEnabled={isBillingEnabled}
        onCreditChange={fetchCredits}
      />
    </SidebarProvider>
  );
}
