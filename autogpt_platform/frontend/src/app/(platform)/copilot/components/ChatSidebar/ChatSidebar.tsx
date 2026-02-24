"use client";
import {
  getGetV2ListSessionsQueryKey,
  useDeleteV2DeleteSession,
  useGetV2ListSessions,
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
  SidebarFooter,
  SidebarHeader,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { DotsThree, PlusCircleIcon, PlusIcon } from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { parseAsString, useQueryState } from "nuqs";
import { useState } from "react";
import { DeleteChatDialog } from "../DeleteChatDialog/DeleteChatDialog";

export function ChatSidebar() {
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  const [sessionToDelete, setSessionToDelete] = useState<{
    id: string;
    title: string | null | undefined;
  } | null>(null);

  const queryClient = useQueryClient();

  const { data: sessionsResponse, isLoading: isLoadingSessions } =
    useGetV2ListSessions({ limit: 50 });

  const { mutate: deleteSession, isPending: isDeleting } =
    useDeleteV2DeleteSession({
      mutation: {
        onSuccess: () => {
          // Invalidate sessions list to refetch
          queryClient.invalidateQueries({
            queryKey: getGetV2ListSessionsQueryKey(),
          });
          // If we deleted the current session, clear selection
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

  const sessions =
    sessionsResponse?.status === 200 ? sessionsResponse.data.sessions : [];

  function handleNewChat() {
    setSessionId(null);
  }

  function handleSelectSession(id: string) {
    setSessionId(id);
  }

  function handleDeleteClick(
    e: React.MouseEvent,
    id: string,
    title: string | null | undefined,
  ) {
    e.stopPropagation(); // Prevent session selection
    if (isDeleting) return; // Prevent double-click during deletion
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
                <Button
                  variant="ghost"
                  onClick={handleNewChat}
                  style={{ minWidth: "auto", width: "auto" }}
                >
                  <PlusCircleIcon className="!size-5" />
                  <span className="sr-only">New Chat</span>
                </Button>
              </div>
            </motion.div>
          </SidebarHeader>
        )}
        <SidebarContent className="gap-4 overflow-y-auto px-4 py-4 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {!isCollapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.2, delay: 0.1 }}
              className="flex items-center justify-between px-3"
            >
              <Text variant="h3" size="body-medium">
                Your chats
              </Text>
              <div className="relative left-6">
                <SidebarTrigger />
              </div>
            </motion.div>
          )}

          {!isCollapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.2, delay: 0.15 }}
              className="mt-4 flex flex-col gap-1"
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
                    <button
                      onClick={() => handleSelectSession(session.id)}
                      className="w-full px-3 py-2.5 pr-10 text-left"
                    >
                      <div className="flex min-w-0 max-w-full flex-col overflow-hidden">
                        <div className="min-w-0 max-w-full">
                          <Text
                            variant="body"
                            className={cn(
                              "truncate font-normal",
                              session.id === sessionId
                                ? "text-zinc-600"
                                : "text-zinc-800",
                            )}
                          >
                            {session.title || `Untitled chat`}
                          </Text>
                        </div>
                        <Text variant="small" className="text-neutral-400">
                          {formatDate(session.updated_at)}
                        </Text>
                      </div>
                    </button>
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
                            handleDeleteClick(e, session.id, session.title)
                          }
                          disabled={isDeleting}
                          className="text-red-600 focus:bg-red-50 focus:text-red-600"
                        >
                          Delete chat
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                ))
              )}
            </motion.div>
          )}
        </SidebarContent>
        {!isCollapsed && sessionId && (
          <SidebarFooter className="shrink-0 bg-zinc-50 p-3 pb-1 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.2, delay: 0.2 }}
            >
              <Button
                variant="primary"
                size="small"
                onClick={handleNewChat}
                className="w-full"
                leftIcon={<PlusIcon className="h-4 w-4" weight="bold" />}
              >
                New Chat
              </Button>
            </motion.div>
          </SidebarFooter>
        )}
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
