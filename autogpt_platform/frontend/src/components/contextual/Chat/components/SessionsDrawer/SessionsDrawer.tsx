"use client";

import { useGetV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import { Text } from "@/components/atoms/Text/Text";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import { X } from "@phosphor-icons/react";
import { formatDistanceToNow } from "date-fns";
import { Drawer } from "vaul";

interface SessionsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectSession: (sessionId: string) => void;
  currentSessionId?: string | null;
}

export function SessionsDrawer({
  isOpen,
  onClose,
  onSelectSession,
  currentSessionId,
}: SessionsDrawerProps) {
  const { data, isLoading } = useGetV2ListSessions(
    { limit: 100 },
    {
      query: {
        enabled: isOpen,
      },
    },
  );

  const sessions =
    data?.status === 200
      ? data.data.sessions.filter((session) => {
          // Filter out sessions without messages (sessions that were never updated)
          // If updated_at equals created_at, the session was created but never had messages
          return session.updated_at !== session.created_at;
        })
      : [];

  function handleSelectSession(sessionId: string) {
    onSelectSession(sessionId);
    onClose();
  }

  return (
    <Drawer.Root
      open={isOpen}
      onOpenChange={(open) => !open && onClose()}
      direction="right"
    >
      <Drawer.Portal>
        <Drawer.Overlay className="fixed inset-0 z-[60] bg-black/10 backdrop-blur-sm" />
        <Drawer.Content
          className={cn(
            "fixed right-0 top-0 z-[70] flex h-full w-96 flex-col border-l border-zinc-200 bg-white",
            scrollbarStyles,
          )}
        >
          <div className="shrink-0 p-4">
            <div className="flex items-center justify-between">
              <Drawer.Title className="text-lg font-semibold">
                Chat Sessions
              </Drawer.Title>
              <button
                aria-label="Close"
                onClick={onClose}
                className="flex size-8 items-center justify-center rounded hover:bg-zinc-100"
              >
                <X width="1.25rem" height="1.25rem" />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Text variant="body" className="text-zinc-500">
                  Loading sessions...
                </Text>
              </div>
            ) : sessions.length === 0 ? (
              <div className="flex h-full items-center justify-center">
                <Text variant="body" className="text-zinc-500">
                  You don&apos;t have previously started chats
                </Text>
              </div>
            ) : (
              <div className="space-y-2">
                {sessions.map((session) => {
                  const isActive = session.id === currentSessionId;
                  const updatedAt = session.updated_at
                    ? formatDistanceToNow(new Date(session.updated_at), {
                        addSuffix: true,
                      })
                    : "";

                  return (
                    <button
                      key={session.id}
                      onClick={() => handleSelectSession(session.id)}
                      className={cn(
                        "w-full rounded-lg border p-3 text-left transition-colors",
                        isActive
                          ? "border-indigo-500 bg-zinc-50"
                          : "border-zinc-200 bg-zinc-100/50 hover:border-zinc-300 hover:bg-zinc-50",
                      )}
                    >
                      <div className="flex flex-col gap-1">
                        <Text
                          variant="body"
                          className={cn(
                            "font-medium",
                            isActive ? "text-indigo-900" : "text-zinc-900",
                          )}
                        >
                          {session.title || "Untitled Chat"}
                        </Text>
                        <div className="flex items-center gap-2 text-xs text-zinc-500">
                          <span>{session.id.slice(0, 8)}...</span>
                          {updatedAt && <span>•</span>}
                          <span>{updatedAt}</span>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}
