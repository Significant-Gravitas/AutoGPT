import { useGetV2GetBuilderIntegrationProvidersInfinite } from "@/app/api/__generated__/endpoints/default/default";
import { ProviderResponse } from "@/app/api/__generated__/models/providerResponse";

const PAGE_SIZE = 10;

export const usePaginatedIntegrationList = () => {
  const {
    data: providers,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: providersLoading,
    error,
    refetch,
  } = useGetV2GetBuilderIntegrationProvidersInfinite(
    {
      page: 1,
      page_size: PAGE_SIZE,
    },
    {
      query: {
        getNextPageParam: (lastPage: any) => {
          const pagination = (lastPage.data as ProviderResponse).pagination;
          const isMore =
            pagination.current_page * pagination.page_size <
            pagination.total_items;

          return isMore ? pagination.current_page + 1 : undefined;
        },
      },
    },
  );

  const allProviders =
    providers?.pages?.flatMap((page: any) => {
      const response = page.data as ProviderResponse;
      return response.providers;
    }) ?? [];

  const status = providers?.pages[0]?.status;

  return {
    allProviders,
    providersLoading,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    error,
    refetch,
    status,
  };
};
