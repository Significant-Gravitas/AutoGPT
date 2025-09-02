import { useInfiniteQuery } from "@tanstack/react-query";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { LibraryAgentResponse } from "@/lib/autogpt-server-api/types";
import { useLibraryPageContext } from "../state-provider";

export const useLibraryAgentList = () => {
  const { searchTerm, librarySort } = useLibraryPageContext();
  const api = useBackendAPI();
  
  const {
    data: agents,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: agentLoading,
  } = useInfiniteQuery({
    queryKey: ["v2", "list", "library", "agents", searchTerm, librarySort],
    queryFn: async ({ pageParam = 1 }) => {
      return await api.listLibraryAgents({
        page: pageParam,
        page_size: 8,
        search_term: searchTerm || undefined,
        sort_by: librarySort,
      });
    },
    getNextPageParam: (lastPage) => {
      const pagination = lastPage.pagination;
      const isMore =
        pagination.current_page * pagination.page_size <
        pagination.total_items;

      return isMore ? pagination.current_page + 1 : undefined;
    },
    initialPageParam: 1,
  });

  const allAgents =
    agents?.pages?.flatMap((page) => {
      return page.agents;
    }) ?? [];

  const agentCount = agents?.pages?.[0]
    ? agents.pages[0].pagination.total_items
    : 0;

  return {
    allAgents,
    agentLoading,
    hasNextPage,
    agentCount,
    isFetchingNextPage,
    fetchNextPage,
  };
};
