"use client";
import {
  getV2GetSession,
  usePatchV2UpdateSessionPinned,
  usePatchV2UpdateSessionTitle,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { Button as ShadcnButton } from "@/components/ui/button";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { ShareChatDialog } from "../../sharing/ShareChatDialog";
import { useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { parseAsString, useQueryState } from "nuqs";
import { useEffect, useRef, useState } from "react";
import { useCopilotChatRuntimeStore } from "../../copilotChatRegistry";
import { formatNotificationTitle } from "../../helpers";
import { fetchAndExportChat } from "../../helpers/exportChatAsMarkdown";
import { shouldShowSessionProcessingIndicator } from "../../sessionActivity";
import { useCopilotUIStore } from "../../store";
import { useSessionDeletion } from "../../useSessionDeletion";
import { useExpertMap } from "../../useExpertMap";
import {
  groupSessionsForSidebar,
  SESSION_LIST_QUERY_KEY,
  useSessionList,
} from "../../useSessionList";
import { useGlobalSearchStore } from "@/app/(platform)/components/GlobalSearchModal/useGlobalSearchStore";
import { useRouter } from "next/navigation";
import { ChatSessionRow } from "./components/ChatSessionRow/ChatSessionRow";
import { DeleteChatDialog } from "../DeleteChatDialog/DeleteChatDialog";
import { UsagePopover } from "../UsageLimits/UsagePopover/UsagePopover";
import { NotificationToggle } from "./components/NotificationToggle/NotificationToggle";
import {
  Files01Icon,
  PlusSignCircleIcon,
  PlusSignIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function ChatSidebar() {
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  const { completedSessionIDs, clearCompletedSession } = useCopilotUIStore();
  const openSearch = useGlobalSearchStore((state) => state.openSearch);
  const closeSearch = useGlobalSearchStore((state) => state.closeSearch);
  const isChatSearchEnabled = useGetFlag(Flag.CHAT_SEARCH);
  const isArtifactsEnabled = useGetFlag(Flag.ARTIFACTS_PAGE);
  const router = useRouter();
  const sessionNeedsReload = useCopilotChatRuntimeStore(
    (state) => state.sessionNeedsReload,
  );

  const queryClient = useQueryClient();

  const {
    sessions,
    isLoading: isLoadingSessions,
    hasMore,
    isLoadingMore,
    loadMore,
  } = useSessionList();

  const {
    sessionToDelete,
    isDeleting,
    requestDelete,
    confirmDelete,
    cancelDelete,
  } = useSessionDeletion();

  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [exportingSessionIds, setExportingSessionIds] = useState<Set<string>>(
    new Set(),
  );
  const [sharingSessionId, setSharingSessionId] = useState<string | null>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);
  const renameCancelledRef = useRef(false);
  const chatSharingEnabled = useGetFlag(Flag.CHAT_SHARING);
  const isPinningEnabled = useGetFlag(Flag.CHAT_PINNING);
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const { expertsById } = useExpertMap();
  const [, setExpertIdParam] = useQueryState("expertId", parseAsString);

  const { mutate: setSessionPinned } = usePatchV2UpdateSessionPinned({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: SESSION_LIST_QUERY_KEY,
        });
      },
      onError: (error) => {
        const description =
          error instanceof ApiError
            ? ((error.response as { detail?: string } | undefined)?.detail ??
              error.message)
            : error instanceof Error
              ? error.message
              : "An error occurred";
        toast({
          title: "Failed to update chat",
          description,
          variant: "destructive",
        });
      },
    },
  });

  const { mutate: renameSession } = usePatchV2UpdateSessionTitle({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: SESSION_LIST_QUERY_KEY,
        });
        setEditingSessionId(null);
      },
      onError: (error) => {
        toast({
          title: "Failed to rename chat",
          description:
            error instanceof Error ? error.message : "An error occurred",
          variant: "destructive",
        });
        setEditingSessionId(null);
      },
    },
  });

  // Auto-focus the rename input when editing starts
  useEffect(() => {
    if (editingSessionId && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [editingSessionId]);

  // Refetch session list when active session changes
  useEffect(() => {
    queryClient.invalidateQueries({
      queryKey: SESSION_LIST_QUERY_KEY,
    });
  }, [sessionId, queryClient]);

  // Clear completed indicator when navigating to a session (works for all paths)
  useEffect(() => {
    if (!sessionId || !completedSessionIDs.has(sessionId)) return;
    clearCompletedSession(sessionId);
    const remaining = Math.max(0, completedSessionIDs.size - 1);
    document.title = formatNotificationTitle(remaining);
  }, [sessionId, completedSessionIDs, clearCompletedSession]);

  function handleNewChat() {
    setSessionId(null);
    // Without this the ?expertId= deep-link adoption in `useChatSession`
    // re-runs on the remount and drops the user straight back into that
    // expert's latest thread, making New Chat a no-op.
    if (isExpertsEnabled) {
      setExpertIdParam(null);
    }
  }

  function handleSelectSession(id: string, expertId: string | null) {
    setSessionId(id);
    if (isExpertsEnabled) {
      setExpertIdParam(expertId);
    }
    closeSearch();
  }

  function handleRenameClick(
    e: React.MouseEvent,
    id: string,
    title: string | null | undefined,
  ) {
    e.stopPropagation();
    renameCancelledRef.current = false;
    setEditingSessionId(id);
    setEditingTitle(title || "");
  }

  function handleRenameSubmit(id: string) {
    const trimmed = editingTitle.trim();
    if (trimmed) {
      renameSession({ sessionId: id, data: { title: trimmed } });
    } else {
      setEditingSessionId(null);
    }
  }

  function handleDeleteClick(
    e: React.MouseEvent,
    id: string,
    title: string | null | undefined,
  ) {
    e.stopPropagation();
    requestDelete(id, title);
  }

  function handlePinClick(e: React.MouseEvent, id: string, isPinned: boolean) {
    e.stopPropagation();
    setSessionPinned({ sessionId: id, data: { is_pinned: !isPinned } });
  }

  async function handleExportClick(
    e: React.MouseEvent,
    id: string,
    title: string | null | undefined,
  ) {
    e.stopPropagation();
    if (exportingSessionIds.has(id)) return;
    setExportingSessionIds((prev) => new Set(prev).add(id));
    try {
      await fetchAndExportChat(id, title, getV2GetSession);
      toast({ title: "Chat exported" });
    } catch (error) {
      console.error("Failed to export chat:", { id, title, error });
      toast({
        title: "Export failed",
        description:
          error instanceof Error
            ? error.message
            : "Could not export this chat. Please try again.",
        variant: "destructive",
      });
    } finally {
      setExportingSessionIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  }

  const { pinned, groups, showHeaders } = groupSessionsForSidebar({
    sessions,
    floatPinned: isPinningEnabled,
  });
  const sessionSections = isExpertsEnabled && showHeaders ? groups : null;

  function renderSessionRow(
    session: SessionSummaryResponse,
    index: number,
    list: SessionSummaryResponse[],
  ) {
    return (
      <ChatSessionRow
        key={session.id}
        session={session}
        isActive={session.id === sessionId}
        isNextActive={list[index + 1]?.id === sessionId}
        isEditing={editingSessionId === session.id}
        editingTitle={editingTitle}
        renameInputRef={renameInputRef}
        isExporting={exportingSessionIds.has(session.id)}
        isDeleting={isDeleting}
        isPinningEnabled={isPinningEnabled}
        isSharingEnabled={chatSharingEnabled}
        showProcessing={
          !!session.is_processing &&
          shouldShowSessionProcessingIndicator({
            sessionId: session.id,
            currentSessionId: sessionId,
            isProcessing: session.is_processing,
            hasCompletedIndicator: completedSessionIDs.has(session.id),
            needsReload: !!sessionNeedsReload[session.id],
          })
        }
        showCompleted={
          completedSessionIDs.has(session.id) && session.id !== sessionId
        }
        onSelect={() =>
          handleSelectSession(session.id, session.expert_id ?? null)
        }
        onEditingTitleChange={setEditingTitle}
        onRenameCancel={() => {
          renameCancelledRef.current = true;
          setEditingSessionId(null);
        }}
        onRenameBlur={() => {
          if (renameCancelledRef.current) {
            renameCancelledRef.current = false;
            return;
          }
          handleRenameSubmit(session.id);
        }}
        onPin={(e) => handlePinClick(e, session.id, !!session.is_pinned)}
        onRename={(e) => handleRenameClick(e, session.id, session.title)}
        onExport={(e) => handleExportClick(e, session.id, session.title)}
        onShare={(e) => {
          e.stopPropagation();
          setSharingSessionId(session.id);
        }}
        onDelete={(e) => handleDeleteClick(e, session.id, session.title)}
      />
    );
  }

  function renderSessionSection(
    key: string,
    label: string,
    list: SessionSummaryResponse[],
  ) {
    const headerId = `session-group-${key}`;
    return (
      <div
        key={key}
        role="group"
        aria-labelledby={headerId}
        className="flex flex-col gap-1"
      >
        <div
          id={headerId}
          data-testid={`expert-group-header-${key}`}
          className="px-3 pb-1 pt-2 text-xs font-semibold uppercase tracking-wide text-zinc-500"
        >
          {label}
        </div>
        {list.map((session, index) => renderSessionRow(session, index, list))}
      </div>
    );
  }

  return (
    <>
      <Sidebar
        variant="inset"
        collapsible="icon"
        className="!top-[calc(50px+var(--preview-banner-height,0px))] !h-[calc(100vh-50px-var(--preview-banner-height,0px))] px-0 [&_[data-sidebar=sidebar]]:border-r [&_[data-sidebar=sidebar]]:border-r-[#80808017]"
      >
        {isCollapsed && (
          <SidebarHeader
            className={cn(
              "flex",
              isCollapsed
                ? "flex-row items-center justify-between gap-y-4 md:flex-col md:items-start md:justify-start"
                : "flex-row items-center justify-between",
            )}
          >
            <motion.div
              key={isCollapsed ? "header-collapsed" : "header-expanded"}
              className="flex flex-col items-center gap-3 pt-4"
              initial={{ opacity: 0, filter: "blur(3px)" }}
              animate={{ opacity: 1, filter: "blur(0px)" }}
              transition={{ type: "spring", bounce: 0.2 }}
            >
              <div className="flex flex-col items-center gap-2">
                <SidebarTrigger />
                {sessionId ? (
                  <Button
                    variant="ghost"
                    onClick={handleNewChat}
                    style={{ minWidth: "auto", width: "auto" }}
                  >
                    <Icon icon={PlusSignCircleIcon} className="!size-5" />
                    <span className="sr-only">New Chat</span>
                  </Button>
                ) : null}
                {isChatSearchEnabled ? (
                  <ShadcnButton
                    type="button"
                    variant="ghost"
                    size="icon-sm"
                    aria-label="Search chats"
                    onClick={() => openSearch()}
                    className="rounded-full text-zinc-600 hover:bg-zinc-100"
                  >
                    <Icon icon={Search01Icon} className="!size-5" />
                  </ShadcnButton>
                ) : null}
              </div>
            </motion.div>
          </SidebarHeader>
        )}
        {!isCollapsed && (
          <SidebarHeader className="shrink-0 px-4 pb-3 pt-3">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.2, delay: 0.1 }}
              className="flex flex-col gap-3 px-3"
            >
              <div className="flex items-center justify-between">
                <Text variant="h3" size="body-medium">
                  Your chats
                </Text>
                <div className="flex items-center">
                  {isChatSearchEnabled ? (
                    <ShadcnButton
                      type="button"
                      variant="ghost"
                      size="icon-sm"
                      aria-label="Search chats"
                      onClick={() => openSearch()}
                      className="rounded-full text-zinc-600 hover:bg-zinc-100"
                    >
                      <Icon icon={Search01Icon} className="!size-5" />
                    </ShadcnButton>
                  ) : null}
                  {isArtifactsEnabled ? (
                    <ShadcnButton
                      type="button"
                      variant="ghost"
                      size="icon-sm"
                      aria-label="Files"
                      onClick={() => router.push("/artifacts")}
                      className="rounded-full text-zinc-600 hover:bg-zinc-100"
                    >
                      <Icon icon={Files01Icon} className="!size-5" />
                    </ShadcnButton>
                  ) : null}
                  <UsagePopover />
                  <NotificationToggle />
                  <SidebarTrigger />
                </div>
              </div>
              {sessionId ? (
                <Button
                  variant="primary"
                  size="small"
                  onClick={handleNewChat}
                  className="w-full"
                  leftIcon={<Icon icon={PlusSignIcon} className="h-4 w-4" />}
                >
                  New Chat
                </Button>
              ) : null}
            </motion.div>
          </SidebarHeader>
        )}

        <SidebarContent className="gap-4 overflow-y-auto px-4 py-4 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {!isCollapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.2, delay: 0.15 }}
              className="flex flex-col gap-1"
            >
              {isLoadingSessions ? (
                <div className="flex min-h-[30rem] items-center justify-center py-4">
                  <LoadingSpinner size="small" className="text-neutral-600" />
                </div>
              ) : !sessions?.length ? (
                <p className="py-4 text-center text-sm text-neutral-500">
                  No conversations yet
                </p>
              ) : sessionSections ? (
                <>
                  {pinned.length > 0 &&
                    renderSessionSection("pinned", "Pinned", pinned)}
                  {sessionSections.map((group) =>
                    renderSessionSection(
                      group.expertId ?? "autopilot",
                      group.expertId
                        ? (expertsById.get(group.expertId)?.name ?? "Expert")
                        : "Autopilot",
                      group.sessions,
                    ),
                  )}
                </>
              ) : (
                sessions.map((session, index) =>
                  renderSessionRow(session, index, sessions),
                )
              )}
              {hasMore && (
                <Button
                  variant="secondary"
                  size="small"
                  onClick={() => loadMore()}
                  loading={isLoadingMore}
                  disabled={isLoadingMore}
                  className="mt-2 w-full"
                >
                  {isLoadingMore ? "Loading…" : "Load older chats"}
                </Button>
              )}
            </motion.div>
          )}
        </SidebarContent>
      </Sidebar>

      <DeleteChatDialog
        session={sessionToDelete}
        isDeleting={isDeleting}
        onConfirm={confirmDelete}
        onCancel={cancelDelete}
      />

      {sharingSessionId && (
        <ShareChatDialog
          sessionId={sharingSessionId}
          open={true}
          onOpenChange={(next) => {
            if (!next) setSharingSessionId(null);
          }}
        />
      )}
    </>
  );
}
