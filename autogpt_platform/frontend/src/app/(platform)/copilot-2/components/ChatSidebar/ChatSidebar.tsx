"use client";
import {
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarFooter,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import {
  SparkleIcon,
  PlusIcon,
  SpinnerGapIcon,
  ChatCircleIcon,
} from "@phosphor-icons/react";
import { motion } from "framer-motion";
import { useState } from "react";
import { parseAsString, useQueryState } from "nuqs";
import {
  postV2CreateSession,
  useGetV2ListSessions,
  getGetV2ListSessionsQueryKey,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { Button } from "@/components/atoms/Button/Button";
import { useQueryClient } from "@tanstack/react-query";

export function ChatSidebar() {
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";
  const [isCreating, setIsCreating] = useState(false);
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  const queryClient = useQueryClient();

  const { data: sessionsResponse, isLoading: isLoadingSessions } =
    useGetV2ListSessions({ limit: 50 });

  const sessions =
    sessionsResponse?.status === 200 ? sessionsResponse.data.sessions : [];

  async function handleNewChat() {
    if (isCreating) return;
    setIsCreating(true);
    try {
      const response = await postV2CreateSession({
        body: JSON.stringify({}),
      });
      if (response.status === 200 && response.data?.id) {
        setSessionId(response.data.id);
        queryClient.invalidateQueries({
          queryKey: getGetV2ListSessionsQueryKey(),
        });
      }
    } finally {
      setIsCreating(false);
    }
  }

  function handleSelectSession(id: string) {
    setSessionId(id);
  }

  function formatDate(dateString: string) {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  }

  return (
    <Sidebar
      variant="inset"
      collapsible="icon"
      className="!top-[60px] !h-[calc(100vh-60px)]"
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
            className={cn(
              "flex items-center gap-2",
              isCollapsed ? "flex-row md:flex-col-reverse" : "flex-row",
            )}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
          >
            {isCollapsed && (
              <div className="h-fit rounded-3xl border border-neutral-400 bg-secondary p-1">
                <SidebarTrigger />
              </div>
            )}
          </motion.div>
        </SidebarHeader>
      )}
      <SidebarContent className="gap-4 overflow-y-auto px-2 py-4 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={handleNewChat}
            disabled={isCreating}
            className={cn(
              "flex h-fit w-full items-center justify-center gap-2 rounded-3xl border-purple-400 bg-purple-100 px-3 py-2 text-purple-600 hover:border-purple-500 hover:bg-purple-200 hover:text-purple-700",
              isCollapsed && "justify-center rounded-3xl px-1",
            )}
          >
            {isCreating ? (
              <SpinnerGapIcon className="h-4 w-4 animate-spin" weight="bold" />
            ) : (
              <PlusIcon className="h-4 w-4" weight="bold" />
            )}
            {!isCollapsed && (
              <span>{isCreating ? "Creating..." : "New Chat"}</span>
            )}
          </Button>
          {!isCollapsed && (
            <div className="h-fit rounded-3xl border border-neutral-400 bg-secondary p-1">
              <SidebarTrigger />
            </div>
          )}
        </div>

        {!isCollapsed && (
          <div className="mt-4 flex flex-col gap-1">
            {isLoadingSessions ? (
              <div className="flex items-center justify-center py-4">
                <SpinnerGapIcon className="h-5 w-5 animate-spin text-neutral-400" />
              </div>
            ) : sessions.length === 0 ? (
              <p className="py-4 text-center text-sm text-neutral-500">
                No conversations yet
              </p>
            ) : (
              sessions.map((session) => (
                <button
                  key={session.id}
                  onClick={() => handleSelectSession(session.id)}
                  className={cn(
                    "flex items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors hover:bg-neutral-100 dark:hover:bg-neutral-800",
                    sessionId === session.id &&
                      "bg-neutral-100 dark:bg-neutral-800",
                  )}
                >
                  <ChatCircleIcon className="h-4 w-4 shrink-0 text-neutral-500" />
                  <div className="flex flex-col overflow-hidden">
                    <span className="truncate font-medium">
                      {session.title || `Untitled chat`}
                    </span>
                    <span className="text-xs text-neutral-500">
                      {formatDate(session.updated_at)}
                    </span>
                  </div>
                </button>
              ))
            )}
          </div>
        )}
      </SidebarContent>
      <SidebarFooter className="px-2"></SidebarFooter>
    </Sidebar>
  );
}
