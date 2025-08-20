import { useGetV1ListGraphExecutionsInfinite } from "@/app/api/__generated__/endpoints/graphs/graphs";
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
        enabled: !!graphID,
        getNextPageParam: (lastPage) => {
          const pagination = (lastPage.data as GraphExecutionsPaginated)
            .pagination;
          const hasMore =
            pagination.current_page * pagination.page_size <
            pagination.total_items;

          return hasMore ? pagination.current_page + 1 : undefined;
        },
      },
    },
    queryClient,
  );

  const agentRuns =
    queryResults?.pages.flatMap((page) => {
      const response = page.data as GraphExecutionsPaginated;
      // FIXME: add reviver function to parse dates coming out of API
      return response.executions;
    }) ?? [];

  const agentRunCount = queryResults?.pages[-1]
    ? (queryResults.pages[-1].data as GraphExecutionsPaginated).pagination
        .total_items
    : 0;

  const upsertAgentRun = (newAgentRun: GraphExecutionMeta) => {
    queryClient.setQueryData(
      [queryKey, { page: 1, page_size: 20 }],
      (currentQueryData: typeof queryResults) => {
        if (!currentQueryData?.pages) return currentQueryData;

        const exists = currentQueryData.pages.some((page) => {
          const response = page.data as GraphExecutionsPaginated;
          return response.executions.some((run) => run.id === newAgentRun.id);
        });
        if (exists) {
          // If the run already exists, we update it
          return {
            ...currentQueryData,
            pages: currentQueryData.pages.map((page) => {
              const response = page.data as GraphExecutionsPaginated;
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
              };
            }),
          };
        }

        // If the run does not exist, we add it to the first page
        const page = currentQueryData.pages[0];
        const updatedExecutions = [
          newAgentRun,
          ...(page.data as GraphExecutionsPaginated).executions,
        ];
        const updatedPage = {
          ...page,
          data: {
            ...page.data,
            executions: updatedExecutions,
          },
        };
        const updatedPages = [updatedPage, ...currentQueryData.pages.slice(1)];
        return {
          ...currentQueryData,
          pages: updatedPages,
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
