import { useInfiniteQuery } from "@tanstack/react-query";
import { BackendAPI } from "@/lib/autogpt-server-api";
import type { LibraryAgentResponse } from "@/lib/autogpt-server-api/types";

export function useFavoriteAgents() {
  const api = new BackendAPI();

  return useInfiniteQuery({
    queryKey: ["favoriteLibraryAgents"],
    queryFn: async ({ pageParam = 1 }) => {
      // Fetch favorite agents from the new endpoint
      const response = await fetch("/api/library/agents/favorites?" + new URLSearchParams({
        page: pageParam.toString(),
        page_size: "10",
      }), {
        headers: {
          "Authorization": `Bearer ${await api.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error("Failed to fetch favorite agents");
      }

      return response.json() as Promise<LibraryAgentResponse>;
    },
    getNextPageParam: (lastPage, pages) => {
      const currentPage = pages.length;
      const totalPages = lastPage.pagination.total_pages;
      return currentPage < totalPages ? currentPage + 1 : undefined;
    },
    initialPageParam: 1,
  });
}