"use client";
import {
  getGetV2ListSessionsQueryKey,
  useDeleteV2DeleteSession,
  useGetV2ListSessions,
  usePatchV2UpdateSessionTitle,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { DotsThree, PlusCircleIcon, PlusIcon } from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { parseAsString, useQueryState } from "nuqs";
import { useEffect, useRef, useState } from "react";
import { getSessionListParams } from "../../helpers";
import { useCopilotUIStore } from "../../store";
import { SessionListItem } from "../SessionListItem/SessionListItem";
import { NotificationToggle } from "./components/NotificationToggle/NotificationToggle";
import { DeleteChatDialog } from "../DeleteChatDialog/DeleteChatDialog";

export function ChatSidebar() {
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  const [showAutopilot, setShowAutopilot] = useQueryState(
    "showAutopilot",
    parseAsString,
  );
  const showAutopilotHistory = showAutopilot === "1";
  const listSessionsParams = getSessionListParams(showAutopilotHistory);
  const {
    sessionToDelete,
    setSessionToDelete,
    completedSessionIDs,
    clearCompletedSession,
  } = useCopilotUIStore();

  const queryClient = useQueryClient();

  const { data: sessionsResponse, isLoading: isLoadingSessions } =
    useGetV2ListSessions(listSessionsParams, {
      query: { refetchInterval: 10_000 },
    });

  const { mutate: deleteSession, isPending: isDeleting } =
    useDeleteV2DeleteSession({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getGetV2ListSessionsQueryKey(),
          });
          if (sessionToDelete?.id === sessionId) {
            setSessionId(null);
          }
          setSessionToDelete(null);
        },
        onError: (error) => {
          toast({
            title: "Failed to delete chat",
            description:
              error instanceof Error ? error.message : "An error occurred",
            variant: "destructive",
          });
          setSessionToDelete(null);
        },
      },
    });

  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);
  const renameCancelledRef = useRef(false);

  const { mutate: renameSession } = usePatchV2UpdateSessionTitle({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetV2ListSessionsQueryKey(),
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
      queryKey: getGetV2ListSessionsQueryKey(),
    });
  }, [sessionId, queryClient]);

  // Clear completed indicator when navigating to a session (works for all paths)
  useEffect(() => {
    if (!sessionId || !completedSessionIDs.has(sessionId)) return;
    clearCompletedSession(sessionId);
    const remaining = completedSessionIDs.size - 1;
    document.title =
      remaining > 0 ? `(${remaining}) Otto is ready - AutoGPT` : "AutoGPT";
  }, [sessionId, completedSessionIDs, clearCompletedSession]);

  const sessions =
    sessionsResponse?.status === 200 ? sessionsResponse.data.sessions : [];

  function handleNewChat() {
    setSessionId(null);
  }

  function handleSelectSession(id: string) {
    setSessionId(id);
  }

  function handleToggleAutopilotHistory() {
    setShowAutopilot(showAutopilotHistory ? null : "1");
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
    if (isDeleting) return;
    setSessionToDelete({ id, title });
  }

  function handleConfirmDelete() {
    if (sessionToDelete) {
      deleteSession({ sessionId: sessionToDelete.id });
    }
  }

  function handleCancelDelete() {
    if (!isDeleting) {
      setSessionToDelete(null);
    }
  }

  return (
    <>
      <Sidebar
        variant="inset"
        collapsible="icon"
        className="!top-[50px] !h-[calc(100vh-50px)] border-r border-zinc-100 px-0"
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
                    <PlusCircleIcon className="!size-5" />
                    <span className="sr-only">New Chat</span>
                  </Button>
                ) : null}
              </div>
            </motion.div>
          </SidebarHeader>
        )}
        {!isCollapsed && (
          <SidebarHeader className="shrink-0 px-4 pb-4 pt-4 shadow-[0_4px_6px_-1px_rgba(0,0,0,0.05)]">
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
                <div className="relative left-5 flex items-center gap-1">
                  <NotificationToggle />
                  <div className="relative left-1">
                    <SidebarTrigger />
                  </div>
                </div>
              </div>
              <div className="flex items-center justify-between gap-3">
                <Text variant="small" className="text-neutral-400">
                  Inspect autopilot sessions
                </Text>
                <Button
                  variant={showAutopilotHistory ? "primary" : "secondary"}
                  size="small"
                  onClick={handleToggleAutopilotHistory}
                  className="min-w-0 px-3 text-xs"
                >
                  {showAutopilotHistory ? "Hide" : "Show"}
                </Button>
              </div>
              {sessionId ? (
                <Button
                  variant="primary"
                  size="small"
                  onClick={handleNewChat}
                  className="w-full"
                  leftIcon={<PlusIcon className="h-4 w-4" weight="bold" />}
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
              ) : sessions.length === 0 ? (
                <p className="py-4 text-center text-sm text-neutral-500">
                  No conversations yet
                </p>
              ) : (
                sessions.map((session) =>
                  editingSessionId === session.id ? (
                    <div
                      key={session.id}
                      className={cn(
                        "group relative w-full rounded-lg transition-colors",
                        session.id === sessionId
                          ? "bg-zinc-100"
                          : "hover:bg-zinc-50",
                      )}
                    >
                      <div className="px-3 py-2.5">
                        <input
                          ref={renameInputRef}
                          type="text"
                          aria-label="Rename chat"
                          value={editingTitle}
                          onChange={(e) => setEditingTitle(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") {
                              e.currentTarget.blur();
                            } else if (e.key === "Escape") {
                              renameCancelledRef.current = true;
                              setEditingSessionId(null);
                            }
                          }}
                          onBlur={() => {
                            if (renameCancelledRef.current) {
                              renameCancelledRef.current = false;
                              return;
                            }
                            handleRenameSubmit(session.id);
                          }}
                          className="w-full rounded border border-zinc-300 bg-white px-2 py-1 text-sm text-zinc-800 outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
                        />
                      </div>
                    </div>
                  ) : (
                    <SessionListItem
                      key={session.id}
                      session={session}
                      currentSessionId={sessionId}
                      isCompleted={completedSessionIDs.has(session.id)}
                      onSelect={handleSelectSession}
                      variant="sidebar"
                      actionSlot={
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <button
                              onClick={(e) => e.stopPropagation()}
                              className="rounded-full p-1.5 text-zinc-600 transition-all hover:bg-neutral-100"
                              aria-label="More actions"
                            >
                              <DotsThree className="h-4 w-4" />
                            </button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem
                              onClick={(e) =>
                                handleRenameClick(e, session.id, session.title)
                              }
                            >
                              Rename
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onClick={(e) =>
                                handleDeleteClick(e, session.id, session.title)
                              }
                              disabled={isDeleting}
                              className="text-red-600 focus:bg-red-50 focus:text-red-600"
                            >
                              Delete chat
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      }
                    />
                  ),
                )
              )}
            </motion.div>
          )}
        </SidebarContent>
      </Sidebar>

      <DeleteChatDialog
        session={sessionToDelete}
        isDeleting={isDeleting}
        onConfirm={handleConfirmDelete}
        onCancel={handleCancelDelete}
      />
    </>
  );
}
