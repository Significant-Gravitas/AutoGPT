import { getPaginationNextPageNumber, unpaginate } from "@/app/api/helpers";
import { useGetV2GetBuilderBlocksInfinite } from "@/app/api/__generated__/endpoints/default/default";

interface UsePaginatedBlocksProps {
  type?: "all" | "input" | "action" | "output" | null;
}

const PAGE_SIZE = 10;
export const usePaginatedBlocks = ({ type }: UsePaginatedBlocksProps) => {
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
      type,
    },
    {
      query: { getNextPageParam: getPaginationNextPageNumber },
    },
  );

  const allBlocks = blocksQueryData
    ? unpaginate(blocksQueryData, "blocks")
    : [];
  const status = blocksQueryData?.pages[0]?.status;

  return {
    allBlocks,
    status,
    blocksLoading,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    error,
    refetch,
  };
};
