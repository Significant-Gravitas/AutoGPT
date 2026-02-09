import { useGetV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { okData } from "@/app/api/helpers";
import { useEffect, useState } from "react";

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

  const { data, isLoading, isFetching, isError } = useGetV2ListSessions(
    { limit: PAGE_SIZE, offset },
    {
      query: {
        enabled: enabled && offset >= 0,
      },
    },
  );

  useEffect(() => {
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
  }, [data, offset, enabled]);

  const hasNextPage =
    totalCount !== null && accumulatedSessions.length < totalCount;

  const areAllSessionsLoaded =
    totalCount !== null &&
    accumulatedSessions.length >= totalCount &&
    !isFetching &&
    !isLoading;

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

  const fetchNextPage = () => {
    if (hasNextPage && !isFetching) {
      setOffset((prev) => prev + PAGE_SIZE);
    }
  };

  const reset = () => {
    // Only reset the offset - keep existing sessions visible during refetch
    // The effect will replace sessions when new data arrives at offset 0
    setOffset(0);
  };

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
