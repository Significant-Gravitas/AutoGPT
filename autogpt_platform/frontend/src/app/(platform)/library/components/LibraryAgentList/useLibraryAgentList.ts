import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import { useLibraryPageContext } from "../state-provider";

export const useLibraryAgentList = () => {
  const { searchTerm, librarySort } = useLibraryPageContext();
  const {
    data: agents,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: agentLoading,
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

  const allAgents =
    agents?.pages?.flatMap((page) => {
      const response = page.data as LibraryAgentResponse;
      return response.agents;
    }) ?? [];

  const agentCount = agents?.pages?.[0]
    ? (agents.pages[0].data as LibraryAgentResponse).pagination.total_items
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
