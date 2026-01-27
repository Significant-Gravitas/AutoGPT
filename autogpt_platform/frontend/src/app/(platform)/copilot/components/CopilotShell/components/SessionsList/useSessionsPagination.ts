import {
  getGetV2ListSessionsQueryKey,
  useGetV2ListSessions,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { okData } from "@/app/api/helpers";
import { useChatStore } from "@/components/contextual/Chat/chat-store";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

const PAGE_SIZE = 50;

export interface UseSessionsPaginationArgs {
  enabled: boolean;
}

export function useSessionsPagination({ enabled }: UseSessionsPaginationArgs) {
  const [offset, setOffset] = useState(0);
  const [accumulatedSessions, setAccumulatedSessions] = useState<
    SessionSummaryResponse[]
  >([]);
  const [totalCount, setTotalCount] = useState<number | null>(null);
  const queryClient = useQueryClient();
  const onStreamComplete = useChatStore((state) => state.onStreamComplete);

  const { data, isLoading, isFetching, isError } = useGetV2ListSessions(
    { limit: PAGE_SIZE, offset },
    {
      query: {
        enabled: enabled && offset >= 0,
      },
    },
  );

  useEffect(
    function refreshOnStreamComplete() {
      const unsubscribe = onStreamComplete(function handleStreamComplete() {
        setOffset(0);
        setAccumulatedSessions([]);
        setTotalCount(null);
        queryClient.invalidateQueries({
          queryKey: getGetV2ListSessionsQueryKey(),
        });
      });
      return unsubscribe;
    },
    [onStreamComplete, queryClient],
  );

  useEffect(
    function updateSessionsFromResponse() {
      const responseData = okData(data);
      if (responseData) {
        const newSessions = responseData.sessions;
        const total = responseData.total;
        setTotalCount(total);

        if (offset === 0) {
          setAccumulatedSessions(newSessions);
        } else {
          setAccumulatedSessions((prev) => [...prev, ...newSessions]);
        }
      } else if (!enabled) {
        setAccumulatedSessions([]);
        setTotalCount(null);
      }
    },
    [data, offset, enabled],
  );

  const hasNextPage = useMemo(() => {
    if (totalCount === null) return false;
    return accumulatedSessions.length < totalCount;
  }, [accumulatedSessions.length, totalCount]);

  const areAllSessionsLoaded = useMemo(() => {
    if (totalCount === null) return false;
    return (
      accumulatedSessions.length >= totalCount && !isFetching && !isLoading
    );
  }, [accumulatedSessions.length, totalCount, isFetching, isLoading]);

  useEffect(() => {
    if (
      hasNextPage &&
      !isFetching &&
      !isLoading &&
      !isError &&
      totalCount !== null
    ) {
      setOffset((prev) => prev + PAGE_SIZE);
    }
  }, [hasNextPage, isFetching, isLoading, isError, totalCount]);

  function fetchNextPage() {
    if (hasNextPage && !isFetching) {
      setOffset((prev) => prev + PAGE_SIZE);
    }
  }

  function reset() {
    setOffset(0);
    setAccumulatedSessions([]);
    setTotalCount(null);
  }

  return {
    sessions: accumulatedSessions,
    isLoading,
    isFetching,
    hasNextPage,
    areAllSessionsLoaded,
    totalCount,
    fetchNextPage,
    reset,
  };
}
