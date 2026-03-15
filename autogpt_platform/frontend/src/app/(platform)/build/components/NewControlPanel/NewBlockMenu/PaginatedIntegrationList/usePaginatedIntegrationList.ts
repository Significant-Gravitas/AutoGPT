import { getPaginationNextPageNumber, unpaginate } from "@/app/api/helpers";
import { useGetV2GetBuilderIntegrationProvidersInfinite } from "@/app/api/__generated__/endpoints/default/default";

const PAGE_SIZE = 10;

export const usePaginatedIntegrationList = () => {
  const {
    data: providersQueryData,
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
      query: { getNextPageParam: getPaginationNextPageNumber },
    },
  );

  const allProviders = providersQueryData
    ? unpaginate(providersQueryData, "providers")
    : [];
  const status = providersQueryData?.pages[0]?.status;

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
