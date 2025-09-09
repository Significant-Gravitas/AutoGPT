import { useInfiniteQuery } from "@tanstack/react-query";
import BackendAPI from "@/lib/autogpt-server-api";
import type { LibraryAgentResponse } from "@/lib/autogpt-server-api/types";

export function useFavoriteAgents() {
  const api = new BackendAPI();

  return useInfiniteQuery({
    queryKey: ["favoriteLibraryAgents"],
    queryFn: async ({ pageParam = 1 }) => {
      // Call the API method to list favorite library agents
      const response = await api.listFavoriteLibraryAgents({
        page: pageParam,
        page_size: 10,
      });
      return response;
    },
    getNextPageParam: (lastPage, pages) => {
      const currentPage = pages.length;
      const totalPages = lastPage.pagination.total_pages;
      return currentPage < totalPages ? currentPage + 1 : undefined;
    },
    initialPageParam: 1,
  });
}
