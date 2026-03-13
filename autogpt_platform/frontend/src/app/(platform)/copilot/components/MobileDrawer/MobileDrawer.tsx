import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import {
  PlusIcon,
  SpeakerHigh,
  SpeakerSlash,
  SpinnerGapIcon,
  X,
} from "@phosphor-icons/react";
import { Drawer } from "vaul";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { useCopilotUIStore } from "../../store";
import { SessionListItem } from "../SessionListItem/SessionListItem";

interface Props {
  isOpen: boolean;
  showAutopilotHistory: boolean;
  sessions: SessionSummaryResponse[];
  currentSessionId: string | null;
  isLoading: boolean;
  onToggleAutopilotHistory: () => void;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
  onClose: () => void;
  onOpenChange: (open: boolean) => void;
}

export function MobileDrawer({
  isOpen,
  showAutopilotHistory,
  sessions,
  currentSessionId,
  isLoading,
  onToggleAutopilotHistory,
  onSelectSession,
  onNewChat,
  onClose,
  onOpenChange,
}: Props) {
  const {
    completedSessionIDs,
    clearCompletedSession,
    isSoundEnabled,
    toggleSound,
  } = useCopilotUIStore();

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
              <div className="flex items-center gap-1">
                <button
                  onClick={toggleSound}
                  className="rounded p-1.5 text-zinc-400 transition-colors hover:text-zinc-600"
                  aria-label={
                    isSoundEnabled
                      ? "Disable notification sound"
                      : "Enable notification sound"
                  }
                >
                  {isSoundEnabled ? (
                    <SpeakerHigh className="h-4 w-4" />
                  ) : (
                    <SpeakerSlash className="h-4 w-4" />
                  )}
                </button>
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
            <div className="mt-2 flex items-center justify-between gap-3">
              <Text variant="small" className="text-neutral-400">
                Inspect autopilot sessions
              </Text>
              <Button
                variant={showAutopilotHistory ? "primary" : "secondary"}
                size="small"
                onClick={onToggleAutopilotHistory}
                className="min-w-0 px-3 text-xs"
              >
                {showAutopilotHistory ? "Hide" : "Show"}
              </Button>
            </div>
            {currentSessionId ? (
              <div className="mt-2">
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
            ) : null}
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
                <SessionListItem
                  key={session.id}
                  session={session}
                  currentSessionId={currentSessionId}
                  isCompleted={completedSessionIDs.has(session.id)}
                  variant="drawer"
                  onSelect={(selectedSessionId) => {
                    onSelectSession(selectedSessionId);
                    if (completedSessionIDs.has(selectedSessionId)) {
                      clearCompletedSession(selectedSessionId);
                    }
                  }}
                />
              ))
            )}
          </div>
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}
