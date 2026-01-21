import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Button } from "@/components/atoms/Button/Button";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import { PlusIcon, X } from "@phosphor-icons/react";
import { Drawer } from "vaul";
import { SessionsList } from "../SessionsList/SessionsList";

interface Props {
  isOpen: boolean;
  sessions: SessionSummaryResponse[];
  currentSessionId: string | null;
  isLoading: boolean;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  onSelectSession: (sessionId: string) => void;
  onFetchNextPage: () => void;
  onNewChat: () => void;
  onClose: () => void;
  onOpenChange: (open: boolean) => void;
  hasActiveSession: boolean;
}

export function MobileDrawer({
  isOpen,
  sessions,
  currentSessionId,
  isLoading,
  hasNextPage,
  isFetchingNextPage,
  onSelectSession,
  onFetchNextPage,
  onNewChat,
  onClose,
  onOpenChange,
  hasActiveSession,
}: Props) {
  return (
    <Drawer.Root open={isOpen} onOpenChange={onOpenChange} direction="left">
      <Drawer.Portal>
        <Drawer.Overlay className="fixed inset-0 z-[60] bg-black/10 backdrop-blur-sm" />
        <Drawer.Content className="fixed left-0 top-0 z-[70] flex h-full w-80 flex-col border-r border-zinc-200 bg-zinc-50">
          <div className="shrink-0 border-b border-zinc-200 p-4">
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
                <X width="1.25rem" height="1.25rem" />
              </Button>
            </div>
          </div>
          <div
            className={cn(
              "flex min-h-0 flex-1 flex-col overflow-y-auto px-3 py-3",
              scrollbarStyles,
            )}
          >
            <SessionsList
              sessions={sessions}
              currentSessionId={currentSessionId}
              isLoading={isLoading}
              hasNextPage={hasNextPage}
              isFetchingNextPage={isFetchingNextPage}
              onSelectSession={onSelectSession}
              onFetchNextPage={onFetchNextPage}
            />
          </div>
          {hasActiveSession && (
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
