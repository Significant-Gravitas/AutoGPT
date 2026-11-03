import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { unpaginate } from "@/app/api/helpers";
import { useEffect, useMemo, useRef } from "react";
import { getNextLibraryPage } from "./helpers";
import { buildAgentInfoMap } from "./store";

const PAGE_SIZE = 100;

export function useLibraryAgents() {
  const {
    data: agentsQueryData,
    isLoading: isRefreshing,
    fetchNextPage,
    refetch,
    hasNextPage,
    isFetching,
    isError,
    isFetchNextPageError,
  } = useGetV2ListLibraryAgentsInfinite(
    {
      page: 1,
      page_size: PAGE_SIZE,
      is_hidden: false,
    },
    {
      query: {
        initialPageParam: 1,
        getNextPageParam: getNextLibraryPage,
        retry: 2,
        refetchOnMount: false,
        refetchOnWindowFocus: false,
        staleTime: 5 * 60 * 1000,
      },
    },
  );

  const attemptedData = useRef<typeof agentsQueryData>(undefined);
  useEffect(() => {
    if (
      hasNextPage &&
      !isFetching &&
      !isError &&
      attemptedData.current !== agentsQueryData
    ) {
      // Cancellation restores the same data; wait for an explicit retry or refresh.
      attemptedData.current = agentsQueryData;
      void fetchNextPage({ cancelRefetch: false });
    }
  }, [agentsQueryData, hasNextPage, isFetching, isError, fetchNextPage]);

  function retry() {
    if (isFetching) return;
    if (isFetchNextPageError || (!isError && hasNextPage)) {
      void fetchNextPage({ cancelRefetch: false });
    } else {
      void refetch({ cancelRefetch: false });
    }
  }

  const agents = useMemo(
    () =>
      agentsQueryData
        ? Array.from(
            new Map(
              unpaginate(agentsQueryData, "agents").map((agent) => [
                agent.id,
                agent,
              ]),
            ).values(),
          )
        : [],
    [agentsQueryData],
  );

  const agentInfoMap = useMemo(() => buildAgentInfoMap(agents), [agents]);

  return {
    agents,
    agentInfoMap,
    isRefreshing,
    isError,
    isFetching,
    retry,
    lastUpdatedAt: undefined,
  };
}
