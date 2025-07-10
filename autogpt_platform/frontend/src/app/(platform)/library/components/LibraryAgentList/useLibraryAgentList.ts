import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import { useScrollThreshold } from "@/hooks/useScrollThreshold";
import { useCallback } from "react";
import { useLibraryPageContext } from "../state-provider";

export const useLibraryAgentList = () => {
  const { searchTerm, librarySort } = useLibraryPageContext();
  const {
    data: agents,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: agentLoading,
    isFetching,
  } = useGetV2ListLibraryAgentsInfinite(
    {
      page: 1,
      page_size: 8,
      search_term: searchTerm || undefined,
      sort_by: librarySort,
    },
    {
      query: {
        getNextPageParam: (lastPage) => {
          const pagination = (lastPage.data as LibraryAgentResponse).pagination;
          const isMore =
            pagination.current_page * pagination.page_size <
            pagination.total_items;

          return isMore ? pagination.current_page + 1 : undefined;
        },
      },
    },
  );

  const handleInfiniteScroll = useCallback(
    (scrollY: number) => {
      if (!hasNextPage || isFetchingNextPage) return;

      const { scrollHeight, clientHeight } = document.documentElement;
      const SCROLL_THRESHOLD = 20;

      if (scrollY + clientHeight >= scrollHeight - SCROLL_THRESHOLD) {
        fetchNextPage();
      }
    },
    [hasNextPage, isFetchingNextPage, fetchNextPage],
  );

  useScrollThreshold(handleInfiniteScroll, 50);

  const allAgents =
    agents?.pages.flatMap((page) => {
      const data = page.data as LibraryAgentResponse;
      return data.agents;
    }) ?? [];

  return {
    allAgents,
    agentLoading,
    isFetchingNextPage,
    hasNextPage,
    isSearching: isFetching && !isFetchingNextPage,
  };
};
