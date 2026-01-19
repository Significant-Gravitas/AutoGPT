import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import { Plus } from "@phosphor-icons/react";
import { SessionsList } from "../SessionsList/SessionsList";

interface Props {
  sessions: SessionSummaryResponse[];
  currentSessionId: string | null;
  isLoading: boolean;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  onSelectSession: (sessionId: string) => void;
  onFetchNextPage: () => void;
  onNewChat: () => void;
}

export function DesktopSidebar({
  sessions,
  currentSessionId,
  isLoading,
  hasNextPage,
  isFetchingNextPage,
  onSelectSession,
  onFetchNextPage,
  onNewChat,
}: Props) {
  return (
    <aside className="flex h-full w-80 flex-col border-r border-zinc-100 bg-white">
      <div className="shrink-0 px-6 py-4">
        <Text variant="h3" size="body-medium">
          Your chats
        </Text>
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
      <div className="shrink-0 bg-white p-3 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
        <Button
          variant="primary"
          size="small"
          onClick={onNewChat}
          className="w-full"
          leftIcon={<Plus width="1rem" height="1rem" />}
        >
          New Chat
        </Button>
      </div>
    </aside>
  );
}
