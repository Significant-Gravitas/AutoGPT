import Link from "next/link";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useSessionList } from "../../copilot/useSessionList";
import { useExpertMap } from "../../copilot/useExpertMap";

export function MobileChats() {
  const {
    sessions,
    isLoading,
    isError,
    refetch,
    hasMore,
    loadMore,
    isLoadingMore,
  } = useSessionList();
  const { expertsById } = useExpertMap();
  return (
    <section aria-label="Your chats" className="flex flex-col gap-4">
      <Link
        href="/home"
        className="flex min-h-11 items-center justify-center rounded-full bg-zinc-900 px-5 text-sm font-medium text-white"
      >
        New chat with Otto
      </Link>
      {isLoading ? (
        <LoadingSpinner />
      ) : isError ? (
        <ErrorCard context="chats" onRetry={() => refetch()} />
      ) : sessions.length === 0 ? (
        <Text variant="body" tone="secondary">
          Your conversations will appear here. Choose an expert or start a chat
          with Otto.
        </Text>
      ) : (
        <ul className="divide-y divide-zinc-100 overflow-hidden rounded-2xl border border-zinc-200 bg-white">
          {sessions.map((session) => (
            <li key={session.id}>
              <Link
                href={`/home?sessionId=${encodeURIComponent(session.id)}`}
                className="flex min-h-20 flex-col justify-center gap-1 px-4 py-3 hover:bg-zinc-50 focus-visible:bg-zinc-50"
              >
                <Text variant="body-medium" unmask={false}>
                  {session.title || "Untitled chat"}
                </Text>
                <Text variant="small" tone="secondary" unmask={false}>
                  {session.expert_id
                    ? (expertsById.get(session.expert_id)?.name ?? "Expert")
                    : "Otto"}
                  {session.is_processing ? " · Working" : ""}
                </Text>
              </Link>
            </li>
          ))}
        </ul>
      )}
      {hasMore && (
        <Button
          variant="secondary"
          onClick={() => loadMore()}
          loading={isLoadingMore}
        >
          Load older chats
        </Button>
      )}
    </section>
  );
}
