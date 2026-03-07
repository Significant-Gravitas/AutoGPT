import { getV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useRef, useState } from "react";
import { convertChatSessionMessagesToUiMessages } from "./helpers/convertChatSessionToUiMessages";

interface UseLoadMoreMessagesArgs {
  sessionId: string | null;
  initialOldestSequence: number | null;
  initialHasMore: boolean;
}

export function useLoadMoreMessages({
  sessionId,
  initialOldestSequence,
  initialHasMore,
}: UseLoadMoreMessagesArgs) {
  const [olderMessages, setOlderMessages] = useState<
    UIMessage<unknown, UIDataTypes, UITools>[]
  >([]);
  const [oldestSequence, setOldestSequence] = useState<number | null>(
    initialOldestSequence,
  );
  const [hasMore, setHasMore] = useState(initialHasMore);
  const [isLoadingMore, setIsLoadingMore] = useState(false);

  // Track the sessionId to reset state on change
  const prevSessionIdRef = useRef(sessionId);

  // Sync initial values from parent when they change
  useEffect(() => {
    if (prevSessionIdRef.current !== sessionId) {
      prevSessionIdRef.current = sessionId;
      setOlderMessages([]);
      setOldestSequence(initialOldestSequence);
      setHasMore(initialHasMore);
      setIsLoadingMore(false);
    } else {
      // Update from parent when initial data changes (e.g. refetch)
      setOldestSequence(initialOldestSequence);
      setHasMore(initialHasMore);
    }
  }, [sessionId, initialOldestSequence, initialHasMore]);

  async function loadMore() {
    if (!sessionId || !hasMore || isLoadingMore || oldestSequence === null)
      return;

    setIsLoadingMore(true);
    try {
      const response = await getV2GetSession(sessionId, {
        limit: 50,
        before_sequence: oldestSequence,
      });

      if (response.status !== 200) return;

      const newMessages = convertChatSessionMessagesToUiMessages(
        sessionId,
        response.data.messages ?? [],
        { isComplete: true },
      );

      setOlderMessages((prev) => [...newMessages, ...prev]);
      setOldestSequence(response.data.oldest_sequence ?? null);
      setHasMore(!!response.data.has_more_messages);
    } finally {
      setIsLoadingMore(false);
    }
  }

  return { olderMessages, hasMore, isLoadingMore, loadMore };
}
