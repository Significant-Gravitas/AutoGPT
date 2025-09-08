import { useGetV2GetBuilderBlocksInfinite } from "@/app/api/__generated__/endpoints/default/default";
import { BlockResponse } from "@/app/api/__generated__/models/blockResponse";
import { useBlockMenuContext } from "../block-menu-provider";

const PAGE_SIZE = 10;

export const useIntegrationBlocks = () => {
  const { integration } = useBlockMenuContext();

  const {
    data: blocks,
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
      query: {
        getNextPageParam: (lastPage) => {
          const pagination = (lastPage.data as BlockResponse).pagination;
          const isMore =
            pagination.current_page * pagination.page_size <
            pagination.total_items;

          return isMore ? pagination.current_page + 1 : undefined;
        },
      },
    },
  );

  const allBlocks =
    blocks?.pages?.flatMap((page) => {
      const response = page.data as BlockResponse;
      return response.blocks;
    }) ?? [];

  const totalBlocks = blocks?.pages[0]
    ? (blocks.pages[0].data as BlockResponse).pagination.total_items
    : 0;

  const status = blocks?.pages[0]?.status;

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
