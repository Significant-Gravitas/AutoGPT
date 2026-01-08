import { InfiniteData, QueryClient } from "@tanstack/react-query";

import {
  getV2ListFavoriteLibraryAgentsResponse,
  getV2ListLibraryAgentsResponse,
} from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

interface UpdateFavoriteInQueriesParams {
  queryClient: QueryClient;
  agentId: string;
  agent: LibraryAgent;
  newIsFavorite: boolean;
}

export function updateFavoriteInQueries({
  queryClient,
  agentId,
  agent,
  newIsFavorite,
}: UpdateFavoriteInQueriesParams) {
  queryClient.setQueriesData(
    { queryKey: ["/api/library/agents"] },
    (
      oldData:
        | InfiniteData<getV2ListLibraryAgentsResponse, number | undefined>
        | undefined,
    ) => {
      if (!oldData?.pages) return oldData;

      return {
        ...oldData,
        pages: oldData.pages.map((page) => {
          if (page.status !== 200) return page;

          return {
            ...page,
            data: {
              ...page.data,
              agents: page.data.agents.map((agent: LibraryAgent) =>
                agent.id === agentId
                  ? { ...agent, is_favorite: newIsFavorite }
                  : agent,
              ),
            },
          };
        }),
      };
    },
  );

  queryClient.setQueriesData(
    { queryKey: ["/api/library/agents/favorites"] },
    (
      oldData:
        | InfiniteData<
            getV2ListFavoriteLibraryAgentsResponse,
            number | undefined
          >
        | undefined,
    ) => {
      if (!oldData?.pages) return oldData;

      if (newIsFavorite) {
        const exists = oldData.pages.some(
          (page) =>
            page.status === 200 &&
            page.data.agents.some(
              (agent: LibraryAgent) => agent.id === agentId,
            ),
        );

        if (!exists) {
          const firstPage = oldData.pages[0];
          if (firstPage?.status === 200) {
            const updatedAgent = {
              id: agent.id,
              name: agent.name,
              description: agent.description,
              graph_id: agent.graph_id,
              can_access_graph: agent.can_access_graph,
              creator_image_url: agent.creator_image_url,
              image_url: agent.image_url,
              is_favorite: true,
            };

            return {
              ...oldData,
              pages: [
                {
                  ...firstPage,
                  data: {
                    ...firstPage.data,
                    agents: [updatedAgent, ...firstPage.data.agents],
                    pagination: {
                      ...firstPage.data.pagination,
                      total_items: firstPage.data.pagination.total_items + 1,
                    },
                  },
                },
                ...oldData.pages.slice(1).map((page) =>
                  page.status === 200
                    ? {
                        ...page,
                        data: {
                          ...page.data,
                          pagination: {
                            ...page.data.pagination,
                            total_items: page.data.pagination.total_items + 1,
                          },
                        },
                      }
                    : page,
                ),
              ],
            };
          }
        }
      } else {
        let removedCount = 0;
        return {
          ...oldData,
          pages: oldData.pages.map((page) => {
            if (page.status !== 200) return page;

            const filteredAgents = page.data.agents.filter(
              (agent: LibraryAgent) => agent.id !== agentId,
            );

            if (filteredAgents.length < page.data.agents.length) {
              removedCount = 1;
            }

            return {
              ...page,
              data: {
                ...page.data,
                agents: filteredAgents,
                pagination: {
                  ...page.data.pagination,
                  total_items: page.data.pagination.total_items - removedCount,
                },
              },
            };
          }),
        };
      }

      return oldData;
    },
  );
}
