"use client";

import { useGetV2ListFavoriteLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";

export function useFavoriteAgents() {
  const {
    data: agents,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: agentLoading,
  } = useGetV2ListFavoriteLibraryAgentsInfinite(
    {
      page: 1,
      page_size: 10,
    },
    {
      query: {
        getNextPageParam: (lastPage) => {
          // Only paginate on successful responses
          if (!lastPage || lastPage.status !== 200) return undefined;

          const pagination = lastPage.data.pagination;
          const isMore =
            pagination.current_page * pagination.page_size <
            pagination.total_items;

          return isMore ? pagination.current_page + 1 : undefined;
        },
      },
    },
  );

  const allAgents =
    agents?.pages?.flatMap((page) => {
      // Only process successful responses
      if (!page || page.status !== 200) return [];
      const response = page.data;
      return response?.agents || [];
    }) ?? [];

  const agentCount = (() => {
    const firstPage = agents?.pages?.[0];
    // Only count from successful responses
    if (!firstPage || firstPage.status !== 200) return 0;
    return firstPage.data?.pagination?.total_items || 0;
  })();

  return {
    allAgents,
    agentLoading,
    hasNextPage,
    agentCount,
    isFetchingNextPage,
    fetchNextPage,
  };
}
