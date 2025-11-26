import {
  getV1ListGraphExecutionsResponse,
  getV1ListGraphExecutionsResponse200,
  useGetV1ListGraphExecutionsInfinite,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { GraphExecutionsPaginated } from "@/app/api/__generated__/models/graphExecutionsPaginated";
import { getQueryClient } from "@/lib/react-query/queryClient";
import {
  GraphExecutionMeta as LegacyGraphExecutionMeta,
  GraphID,
  GraphExecutionID,
} from "@/lib/autogpt-server-api";
import { GraphExecutionMeta as RawGraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";

export type GraphExecutionMeta = Omit<
  RawGraphExecutionMeta,
  "id" | "user_id" | "graph_id" | "preset_id" | "stats"
> &
  Pick<
    LegacyGraphExecutionMeta,
    "id" | "user_id" | "graph_id" | "preset_id" | "stats"
  >;

/** Hook to fetch runs for a specific graph, with support for infinite scroll.
 *
 * @param graphID - The ID of the graph to fetch agent runs for. This parameter is
 *                  optional in the sense that the hook doesn't run unless it is passed.
 *                  This way, it can be used in components where the graph ID is not
 *                  immediately available.
 */
export const useAgentRunsInfinite = (graphID?: GraphID) => {
  const queryClient = getQueryClient();
  const {
    data: queryResults,
    refetch: refetchRuns,
    isPending: agentRunsLoading,
    isRefetching: agentRunsReloading,
    hasNextPage: hasMoreRuns,
    fetchNextPage: fetchMoreRuns,
    isFetchingNextPage: isFetchingMoreRuns,
    queryKey,
  } = useGetV1ListGraphExecutionsInfinite(
    graphID!,
    { page: 1, page_size: 20 },
    {
      query: {
        getNextPageParam: (lastPage) => {
          const pagination = (lastPage.data as GraphExecutionsPaginated)
            .pagination;
          const hasMore =
            pagination.current_page * pagination.page_size <
            pagination.total_items;

          return hasMore ? pagination.current_page + 1 : undefined;
        },

        // Prevent query from running if graphID is not available (yet)
        ...(!graphID
          ? {
              enabled: false,
              queryFn: () =>
                // Fake empty response if graphID is not available (yet)
                Promise.resolve({
                  status: 200,
                  data: {
                    executions: [],
                    pagination: {
                      current_page: 1,
                      page_size: 20,
                      total_items: 0,
                      total_pages: 0,
                    },
                  },
                  headers: new Headers(),
                } satisfies getV1ListGraphExecutionsResponse),
            }
          : {}),
      },
    },
    queryClient,
  );

  const agentRuns =
    queryResults?.pages.flatMap((page) => {
      const response = page.data as GraphExecutionsPaginated;
      return response.executions;
    }) ?? [];

  const agentRunCount = (
    queryResults?.pages.at(-1)?.data as GraphExecutionsPaginated | undefined
  )?.pagination.total_items;

  const upsertAgentRun = (newAgentRun: GraphExecutionMeta) => {
    queryClient.setQueryData(
      queryKey,
      (currentQueryData: typeof queryResults) => {
        if (!currentQueryData?.pages || agentRunCount === undefined)
          return currentQueryData;

        const exists = currentQueryData.pages.some((page) => {
          if (page.status !== 200) return false;

          const response = page.data;
          return response.executions.some((run) => run.id === newAgentRun.id);
        });
        if (exists) {
          // If the run already exists, we update it
          return {
            ...currentQueryData,
            pages: currentQueryData.pages.map((page) => {
              if (page.status !== 200) return page;
              const response = page.data;
              const executions = response.executions;

              const index = executions.findIndex(
                (run) => run.id === newAgentRun.id,
              );
              if (index === -1) return page;

              const newExecutions = [...executions];
              newExecutions[index] = newAgentRun;

              return {
                ...page,
                data: {
                  ...response,
                  executions: newExecutions,
                },
              } satisfies getV1ListGraphExecutionsResponse;
            }),
          };
        }

        // If the run does not exist, we add it to the first page
        const page = currentQueryData
          .pages[0] as getV1ListGraphExecutionsResponse200 & {
          headers: Headers;
        };
        const updatedExecutions = [newAgentRun, ...page.data.executions];
        const updatedPage = {
          ...page,
          data: {
            ...page.data,
            executions: updatedExecutions,
          },
        } satisfies getV1ListGraphExecutionsResponse;
        const updatedPages = [updatedPage, ...currentQueryData.pages.slice(1)];
        return {
          ...currentQueryData,
          pages: updatedPages.map(
            // Increment the total runs count in the pagination info of all pages
            (page) =>
              page.status === 200
                ? {
                    ...page,
                    data: {
                      ...page.data,
                      pagination: {
                        ...page.data.pagination,
                        total_items: agentRunCount + 1,
                      },
                    },
                  }
                : page,
          ),
        };
      },
    );
  };

  const removeAgentRun = (runID: GraphExecutionID) => {
    queryClient.setQueryData(
      [queryKey, { page: 1, page_size: 20 }],
      (currentQueryData: typeof queryResults) => {
        if (!currentQueryData?.pages) return currentQueryData;

        let found = false;
        return {
          ...currentQueryData,
          pages: currentQueryData.pages.map((page) => {
            const response = page.data as GraphExecutionsPaginated;
            const filteredExecutions = response.executions.filter(
              (run) => run.id !== runID,
            );
            if (filteredExecutions.length < response.executions.length) {
              found = true;
            }

            return {
              ...page,
              data: {
                ...response,
                executions: filteredExecutions,
                pagination: {
                  ...response.pagination,
                  total_items:
                    response.pagination.total_items - (found ? 1 : 0),
                },
              },
            };
          }),
        };
      },
    );
  };

  return {
    agentRuns: agentRuns as GraphExecutionMeta[],
    refetchRuns,
    agentRunCount,
    agentRunsLoading: agentRunsLoading || agentRunsReloading,
    hasMoreRuns,
    fetchMoreRuns,
    isFetchingMoreRuns,
    upsertAgentRun,
    removeAgentRun,
  };
};

export type AgentRunsQuery = ReturnType<typeof useAgentRunsInfinite>;
