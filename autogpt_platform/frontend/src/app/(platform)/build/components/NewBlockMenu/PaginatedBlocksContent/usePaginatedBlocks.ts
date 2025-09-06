import { useInfiniteQuery } from "@tanstack/react-query";
import { getV2GetBuilderBlocks, useGetV2GetBuilderBlocksInfinite } from "@/app/api/__generated__/endpoints/default/default";
import { BlockResponse } from "@/app/api/__generated__/models/blockResponse";
import { GetV2GetBuilderBlocksParams } from "@/app/api/__generated__/models/getV2GetBuilderBlocksParams";
import { BlockType } from "../BlockList";

interface UsePaginatedBlocksProps {
  category?: string | null;
  type?: "all" | "input" | "action" | "output" | null;
  provider?: string | null;
  pageSize?: number;
}

export const usePaginatedBlocks = ({
  type,
  pageSize = 10,
}: UsePaginatedBlocksProps) => {
  const {data: blocks,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: blocksLoading,} = useGetV2GetBuilderBlocksInfinite({
    page: 1,
    page_size: pageSize,
    type,
  },{
    query: {
      getNextPageParam: (lastPage) => {
        const pagination = (lastPage.data as BlockResponse).pagination;
        const isMore =
          pagination.current_page * pagination.page_size <
          pagination.total_items;

        return isMore ? pagination.current_page + 1 : undefined;
      },
    },
  },)  

  const allBlocks = blocks?.pages?.flatMap((page) => {
    const response = page.data as BlockResponse;
    return response.blocks;
  }) ?? [];

  return {
    allBlocks,
    blocksLoading,
    hasNextPage,

    isFetchingNextPage,
    fetchNextPage,
  };
};
