import { useBlockMenuStore } from "../../../../stores/blockMenuStore";
import { useGetV2BuilderSearchInfinite } from "@/app/api/__generated__/endpoints/store/store";
import { SearchResponse } from "@/app/api/__generated__/models/searchResponse";

export const useBlockMenuSearch = () => {
  const { searchQuery } = useBlockMenuStore();

  const {
    data: searchData,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: searchLoading,
  } = useGetV2BuilderSearchInfinite(
    {
      page: 1,
      page_size: 8,
      search_query: searchQuery,
    },
    {
      query: {
        getNextPageParam: (lastPage, allPages) => {
          const pagination = lastPage.data as SearchResponse;
          const isMore = pagination.more_pages;
          return isMore ? allPages.length + 1 : undefined;
        },
      },
    },
  );

  const allSearchData =
    searchData?.pages?.flatMap((page) => {
      const response = page.data as SearchResponse;
      return response.items;
    }) ?? [];

  return {
    allSearchData,
    isFetchingNextPage,
    fetchNextPage,
    hasNextPage,
    searchLoading,
  };
};
