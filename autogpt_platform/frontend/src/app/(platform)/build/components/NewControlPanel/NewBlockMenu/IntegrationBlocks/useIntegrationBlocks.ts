import {
  getPaginatedTotalCount,
  getPaginationNextPageNumber,
  unpaginate,
} from "@/app/api/helpers";
import { useGetV2GetBuilderBlocksInfinite } from "@/app/api/__generated__/endpoints/default/default";
import { useBlockMenuStore } from "@/app/(platform)/build/stores/blockMenuStore";

const PAGE_SIZE = 10;

export const useIntegrationBlocks = () => {
  const { integration } = useBlockMenuStore();

  const {
    data: blocksQueryData,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: blocksLoading,
    error,
    refetch,
  } = useGetV2GetBuilderBlocksInfinite(
    {
      page: 1,
      page_size: PAGE_SIZE,
      provider: integration,
    },
    {
      query: { getNextPageParam: getPaginationNextPageNumber },
    },
  );

  const allBlocks = blocksQueryData
    ? unpaginate(blocksQueryData, "blocks")
    : [];
  const totalBlocks = getPaginatedTotalCount(blocksQueryData);

  const status = blocksQueryData?.pages[0]?.status;

  return {
    allBlocks,
    totalBlocks,
    status,
    blocksLoading,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    error,
    refetch,
  };
};
