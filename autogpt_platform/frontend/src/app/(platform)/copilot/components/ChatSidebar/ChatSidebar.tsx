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
import {
  CheckCircle,
  DotsThree,
  PlusCircleIcon,
  PlusIcon,
} from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { parseAsString, useQueryState } from "nuqs";
import { useEffect, useRef, useState } from "react";
import { useCopilotUIStore } from "../../store";
import { NotificationToggle } from "./components/NotificationToggle/NotificationToggle";
import { DeleteChatDialog } from "../DeleteChatDialog/DeleteChatDialog";
import { PulseLoader } from "../PulseLoader/PulseLoader";
import { UsageLimits } from "../UsageLimits/UsageLimits";

export function ChatSidebar() {
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  const {
    sessionToDelete,
    setSessionToDelete,
    completedSessionIDs,
    clearCompletedSession,
  } = useCopilotUIStore();

  const queryClient = useQueryClient();

  const { data: sessionsResponse, isLoading: isLoadingSessions } =
    useGetV2ListSessions({ limit: 50 }, { query: { refetchInterval: 10_000 } });

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

  function formatDate(dateString: string) {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return `${diffDays} days ago`;

    const day = date.getDate();
    const ordinal =
      day % 10 === 1 && day !== 11
        ? "st"
        : day % 10 === 2 && day !== 12
          ? "nd"
          : day % 10 === 3 && day !== 13
            ? "rd"
            : "th";
    const month = date.toLocaleDateString("en-US", { month: "short" });
    const year = date.getFullYear();

    return `${day}${ordinal} ${month} ${year}`;
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
                <div className="flex items-center">
                  <UsageLimits />
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
                sessions.map((session) => (
                  <div
                    key={session.id}
                    className={cn(
                      "group relative w-full rounded-lg transition-colors",
                      session.id === sessionId
                        ? "bg-zinc-100"
                        : "hover:bg-zinc-50",
                    )}
                  >
                    {editingSessionId === session.id ? (
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
                    ) : (
                      <button
                        onClick={() => handleSelectSession(session.id)}
                        className="w-full px-3 py-2.5 pr-10 text-left"
                      >
                        <div className="flex min-w-0 max-w-full items-center gap-2">
                          <div className="min-w-0 flex-1">
                            <Text
                              variant="body"
                              className={cn(
                                "truncate font-normal",
                                session.id === sessionId
                                  ? "text-zinc-600"
                                  : "text-zinc-800",
                              )}
                            >
                              <AnimatePresence mode="wait" initial={false}>
                                <motion.span
                                  key={session.title || "untitled"}
                                  initial={{ opacity: 0, y: 4 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  exit={{ opacity: 0, y: -4 }}
                                  transition={{ duration: 0.2 }}
                                  className="block truncate"
                                >
                                  {session.title || "Untitled chat"}
                                </motion.span>
                              </AnimatePresence>
                            </Text>
                            <Text variant="small" className="text-neutral-400">
                              {formatDate(session.updated_at)}
                            </Text>
                          </div>
                          {session.is_processing &&
                            session.id !== sessionId &&
                            !completedSessionIDs.has(session.id) && (
                              <PulseLoader size={16} className="shrink-0" />
                            )}
                          {completedSessionIDs.has(session.id) &&
                            session.id !== sessionId && (
                              <CheckCircle
                                className="h-4 w-4 shrink-0 text-green-500"
                                weight="fill"
                              />
                            )}
                        </div>
                      </button>
                    )}
                    {editingSessionId !== session.id && (
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <button
                            onClick={(e) => e.stopPropagation()}
                            className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full p-1.5 text-zinc-600 transition-all hover:bg-neutral-100"
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
                    )}
                  </div>
                ))
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
