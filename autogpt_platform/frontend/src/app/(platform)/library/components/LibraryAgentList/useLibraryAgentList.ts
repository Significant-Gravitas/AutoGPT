"use client";

import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import { useLibraryPageContext } from "../state-provider";
import { useLibraryAgentsStore } from "@/hooks/useLibraryAgents/store";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { getV2ListLibraryAgentsResponse } from "@/app/api/__generated__/endpoints/library/library";

export const useLibraryAgentList = () => {
  const { searchTerm, librarySort } = useLibraryPageContext();
  const { agents: cachedAgents } = useLibraryAgentsStore();

  function filterAgents(agents: LibraryAgent[], term?: string | null) {
    const t = term?.trim().toLowerCase();
    if (!t) return agents;
    return agents.filter(
      (a) =>
        a.name.toLowerCase().includes(t) ||
        a.description.toLowerCase().includes(t),
    );
  }

  function getInitialData(pageSize: number) {
    const filtered = filterAgents(cachedAgents as LibraryAgent[], searchTerm);
    if (!filtered.length) return undefined;

    const firstPageAgents = filtered.slice(0, pageSize);
    const totalItems = filtered.length;
    const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));

    const firstPage: getV2ListLibraryAgentsResponse = {
      status: 200,
      data: {
        agents: firstPageAgents,
        pagination: {
          total_items: totalItems,
          total_pages: totalPages,
          current_page: 1,
          page_size: pageSize,
        },
      } as LibraryAgentResponse,
      headers: new Headers(),
    };

    return { pageParams: [1], pages: [firstPage] };
  }

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
        initialData: getInitialData(8),
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
