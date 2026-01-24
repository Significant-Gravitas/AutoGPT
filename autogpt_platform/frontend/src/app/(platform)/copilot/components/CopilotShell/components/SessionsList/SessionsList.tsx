import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { InfiniteList } from "@/components/molecules/InfiniteList/InfiniteList";
import { cn } from "@/lib/utils";
import { getSessionTitle } from "../../helpers";

interface Props {
  sessions: SessionSummaryResponse[];
  currentSessionId: string | null;
  isLoading: boolean;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  onSelectSession: (sessionId: string) => void;
  onFetchNextPage: () => void;
}

export function SessionsList({
  sessions,
  currentSessionId,
  isLoading,
  hasNextPage,
  isFetchingNextPage,
  onSelectSession,
  onFetchNextPage,
}: Props) {
  if (isLoading) {
    return (
      <div className="space-y-1">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="rounded-lg px-3 py-2.5">
            <Skeleton className="h-5 w-full" />
          </div>
        ))}
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <Text variant="body" className="text-zinc-500">
          You don&apos;t have previous chats
        </Text>
      </div>
    );
  }

  return (
    <InfiniteList
      items={sessions}
      hasMore={hasNextPage}
      isFetchingMore={isFetchingNextPage}
      onEndReached={onFetchNextPage}
      className="space-y-1"
      renderItem={(session) => {
        const isActive = session.id === currentSessionId;
        return (
          <button
            onClick={() => onSelectSession(session.id)}
            className={cn(
              "w-full rounded-lg px-3 py-2.5 text-left transition-colors",
              isActive ? "bg-zinc-100" : "hover:bg-zinc-50",
            )}
          >
            <Text
              variant="body"
              className={cn(
                "font-normal",
                isActive ? "text-zinc-600" : "text-zinc-800",
              )}
            >
              {getSessionTitle(session)}
            </Text>
          </button>
        );
      }}
    />
  );
}
