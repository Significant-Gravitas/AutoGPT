import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import { PlusIcon, SpinnerGapIcon, X } from "@phosphor-icons/react";
import { Drawer } from "vaul";

interface Props {
  isOpen: boolean;
  sessions: SessionSummaryResponse[];
  currentSessionId: string | null;
  isLoading: boolean;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
  onClose: () => void;
  onOpenChange: (open: boolean) => void;
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

export function MobileDrawer({
  isOpen,
  sessions,
  currentSessionId,
  isLoading,
  onSelectSession,
  onNewChat,
  onClose,
  onOpenChange,
}: Props) {
  return (
    <Drawer.Root open={isOpen} onOpenChange={onOpenChange} direction="left">
      <Drawer.Portal>
        <Drawer.Overlay className="fixed inset-0 z-[60] bg-black/10 backdrop-blur-sm" />
        <Drawer.Content className="fixed left-0 top-0 z-[70] flex h-full w-80 flex-col border-r border-zinc-200 bg-zinc-50">
          <div className="shrink-0 border-b border-zinc-200 px-4 py-2">
            <div className="flex items-center justify-between">
              <Drawer.Title className="text-lg font-semibold text-zinc-800">
                Your chats
              </Drawer.Title>
              <Button
                variant="icon"
                size="icon"
                aria-label="Close sessions"
                onClick={onClose}
              >
                <X width="1rem" height="1rem" />
              </Button>
            </div>
          </div>
          <div
            className={cn(
              "flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto px-3 py-3",
              scrollbarStyles,
            )}
          >
            {isLoading ? (
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
                  onClick={() => onSelectSession(session.id)}
                  className={cn(
                    "w-full rounded-lg px-3 py-2.5 text-left transition-colors",
                    session.id === currentSessionId
                      ? "bg-zinc-100"
                      : "hover:bg-zinc-50",
                  )}
                >
                  <div className="flex min-w-0 max-w-full flex-col overflow-hidden">
                    <div className="min-w-0 max-w-full">
                      <Text
                        variant="body"
                        className={cn(
                          "truncate font-normal",
                          session.id === currentSessionId
                            ? "text-zinc-600"
                            : "text-zinc-800",
                        )}
                      >
                        {session.title || "Untitled chat"}
                      </Text>
                    </div>
                    <Text variant="small" className="text-neutral-400">
                      {formatDate(session.updated_at)}
                    </Text>
                  </div>
                </button>
              ))
            )}
          </div>
          {currentSessionId && (
            <div className="shrink-0 bg-white p-3 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
              <Button
                variant="primary"
                size="small"
                onClick={onNewChat}
                className="w-full"
                leftIcon={<PlusIcon width="1rem" height="1rem" />}
              >
                New Chat
              </Button>
            </div>
          )}
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}
