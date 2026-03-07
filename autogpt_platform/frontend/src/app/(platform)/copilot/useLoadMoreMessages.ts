import { getV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useRef, useState } from "react";
import { convertChatSessionMessagesToUiMessages } from "./helpers/convertChatSessionToUiMessages";

interface UseLoadMoreMessagesArgs {
  sessionId: string | null;
  initialOldestSequence: number | null;
  initialHasMore: boolean;
}

const MAX_CONSECUTIVE_ERRORS = 3;

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
  const isLoadingMoreRef = useRef(false);
  const consecutiveErrorsRef = useRef(0);

  // Track the sessionId and initial cursor to reset state on change
  const prevSessionIdRef = useRef(sessionId);
  const prevInitialOldestRef = useRef(initialOldestSequence);

  // Sync initial values from parent when they change
  useEffect(() => {
    if (prevSessionIdRef.current !== sessionId) {
      // Session changed — full reset
      prevSessionIdRef.current = sessionId;
      prevInitialOldestRef.current = initialOldestSequence;
      setOlderMessages([]);
      setOldestSequence(initialOldestSequence);
      setHasMore(initialHasMore);
      setIsLoadingMore(false);
      isLoadingMoreRef.current = false;
      consecutiveErrorsRef.current = 0;
    } else if (
      prevInitialOldestRef.current !== initialOldestSequence &&
      olderMessages.length > 0
    ) {
      // Same session but initial window shifted (e.g. new messages arrived) —
      // clear paged state to avoid gaps/duplicates
      prevInitialOldestRef.current = initialOldestSequence;
      setOlderMessages([]);
      setOldestSequence(initialOldestSequence);
      setHasMore(initialHasMore);
      setIsLoadingMore(false);
      isLoadingMoreRef.current = false;
      consecutiveErrorsRef.current = 0;
    } else {
      // Update from parent when initial data changes (e.g. refetch)
      prevInitialOldestRef.current = initialOldestSequence;
      setOldestSequence(initialOldestSequence);
      setHasMore(initialHasMore);
    }
  }, [sessionId, initialOldestSequence, initialHasMore]);

  async function loadMore() {
    if (
      !sessionId ||
      !hasMore ||
      isLoadingMoreRef.current ||
      oldestSequence === null
    )
      return;

    isLoadingMoreRef.current = true;
    setIsLoadingMore(true);
    try {
      const response = await getV2GetSession(sessionId, {
        limit: 50,
        before_sequence: oldestSequence,
      });

      if (response.status !== 200) {
        consecutiveErrorsRef.current += 1;
        console.warn(
          `[loadMore] Failed to load messages (status=${response.status}, attempt=${consecutiveErrorsRef.current})`,
        );
        if (consecutiveErrorsRef.current >= MAX_CONSECUTIVE_ERRORS) {
          setHasMore(false);
        }
        return;
      }

      consecutiveErrorsRef.current = 0;

      const newMessages = convertChatSessionMessagesToUiMessages(
        sessionId,
        response.data.messages ?? [],
        { isComplete: true },
      );

      setOlderMessages((prev) => [...newMessages, ...prev]);
      setOldestSequence(response.data.oldest_sequence ?? null);
      setHasMore(!!response.data.has_more_messages);
    } catch (error) {
      consecutiveErrorsRef.current += 1;
      console.warn("[loadMore] Network error:", error);
      if (consecutiveErrorsRef.current >= MAX_CONSECUTIVE_ERRORS) {
        setHasMore(false);
      }
    } finally {
      isLoadingMoreRef.current = false;
      setIsLoadingMore(false);
    }
  }

  return { olderMessages, hasMore, isLoadingMore, loadMore };
}
