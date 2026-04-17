import { getV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  convertChatSessionMessagesToUiMessages,
  extractToolOutputsFromRaw,
} from "./helpers/convertChatSessionToUiMessages";

interface UseLoadMoreMessagesArgs {
  sessionId: string | null;
  initialOldestSequence: number | null;
  initialHasMore: boolean;
  /** Raw messages from the initial page, used for cross-page tool output matching. */
  initialPageRawMessages: unknown[];
}

const MAX_CONSECUTIVE_ERRORS = 3;
const MAX_OLDER_MESSAGES = 2000;

export function useLoadMoreMessages({
  sessionId,
  initialOldestSequence,
  initialHasMore,
  initialPageRawMessages,
}: UseLoadMoreMessagesArgs) {
  // Accumulated raw messages from all extra pages (ascending order).
  // Re-converting them all together ensures tool outputs are matched across
  // inter-page boundaries.
  const [pagedRawMessages, setPagedRawMessages] = useState<unknown[]>([]);
  const [oldestSequence, setOldestSequence] = useState<number | null>(
    initialOldestSequence,
  );
  const [hasMore, setHasMore] = useState(initialHasMore);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const isLoadingMoreRef = useRef(false);
  const consecutiveErrorsRef = useRef(0);
  // Epoch counter to discard stale loadMore responses after a reset
  const epochRef = useRef(0);

  const prevSessionIdRef = useRef(sessionId);

  // Sync initial values from parent when they change.
  //
  // The parent's `initialOldestSequence` drifts forward every time the
  // session query refetches (e.g. after a stream completes — see
  // `useCopilotStream` invalidation on `streaming → ready`). If we
  // wiped `pagedRawMessages` every time that happened, users who had
  // scrolled back would lose their loaded history on each new turn and
  // subsequent `loadMore` calls would fetch messages that overlap with
  // the AI SDK's retained state in `currentMessages`, producing visible
  // duplicates.
  //
  // Instead: once any older page is loaded, preserve local state across
  // refetches. The local cursor (`oldestSequence`) still points to the
  // oldest message we've explicitly loaded, so the next `loadMore`
  // fetches cleanly before it. Any messages between the refetched
  // initial window and the older pages are covered by AI SDK's
  // retained state in `currentMessages`.
  useEffect(() => {
    if (prevSessionIdRef.current !== sessionId) {
      // Session changed — full reset
      prevSessionIdRef.current = sessionId;
      setPagedRawMessages([]);
      setOldestSequence(initialOldestSequence);
      setHasMore(initialHasMore);
      setIsLoadingMore(false);
      isLoadingMoreRef.current = false;
      consecutiveErrorsRef.current = 0;
      epochRef.current += 1;
      return;
    }

    // If we haven't paged yet, mirror the parent so the first
    // `loadMore` starts from the correct cursor.
    if (pagedRawMessages.length === 0) {
      setOldestSequence(initialOldestSequence);
      setHasMore(initialHasMore);
    }
  }, [sessionId, initialOldestSequence, initialHasMore]);

  // Convert all accumulated raw messages in one pass so tool outputs
  // are matched across inter-page boundaries.
  // Include initial page tool outputs so older paged pages can match
  // tool calls whose outputs landed in the initial page.
  const pagedMessages: UIMessage<unknown, UIDataTypes, UITools>[] =
    useMemo(() => {
      if (!sessionId || pagedRawMessages.length === 0) return [];
      const extraToolOutputs =
        initialPageRawMessages.length > 0
          ? extractToolOutputsFromRaw(initialPageRawMessages)
          : undefined;
      return convertChatSessionMessagesToUiMessages(
        sessionId,
        pagedRawMessages,
        { isComplete: true, extraToolOutputs },
      ).messages;
    }, [sessionId, pagedRawMessages, initialPageRawMessages]);

  async function loadMore() {
    if (!sessionId || !hasMore || isLoadingMoreRef.current) return;
    if (oldestSequence === null) return;

    const requestEpoch = epochRef.current;
    isLoadingMoreRef.current = true;
    setIsLoadingMore(true);
    try {
      const response = await getV2GetSession(sessionId, {
        limit: 50,
        before_sequence: oldestSequence,
      });

      // Discard response if session/pagination was reset while awaiting
      if (epochRef.current !== requestEpoch) return;

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

      const newRaw = (response.data.messages ?? []) as unknown[];
      const estimatedTotal = pagedRawMessages.length + newRaw.length;
      setPagedRawMessages((prev) => {
        const merged = [...newRaw, ...prev];
        if (merged.length > MAX_OLDER_MESSAGES) {
          return merged.slice(merged.length - MAX_OLDER_MESSAGES);
        }
        return merged;
      });

      // Note: after truncation, oldest_sequence may reference a dropped
      // message. This is safe because we also set hasMore=false below,
      // preventing further loads with the stale cursor.
      setOldestSequence(response.data.oldest_sequence ?? null);
      if (estimatedTotal >= MAX_OLDER_MESSAGES) {
        setHasMore(false);
      } else {
        setHasMore(!!response.data.has_more_messages);
      }
    } catch (error) {
      if (epochRef.current !== requestEpoch) return;
      consecutiveErrorsRef.current += 1;
      console.warn("[loadMore] Network error:", error);
      if (consecutiveErrorsRef.current >= MAX_CONSECUTIVE_ERRORS) {
        setHasMore(false);
      }
    } finally {
      if (epochRef.current === requestEpoch) {
        isLoadingMoreRef.current = false;
        setIsLoadingMore(false);
      }
    }
  }

  return { pagedMessages, hasMore, isLoadingMore, loadMore };
}
