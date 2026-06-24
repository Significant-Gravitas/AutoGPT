import { SearchResponseItemsItem } from "@/app/api/__generated__/models/searchResponseItemsItem";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { getSearchItemType } from "./helper";
import { MarketplaceAgentBlock } from "../MarketplaceAgentBlock";
import { Block } from "../Block";
import { UGCAgentBlock } from "../UGCAgentBlock";
import { useBlockMenuSearchContent } from "./useBlockMenuSearchContent";
import { useBlockMenuStore } from "@/app/(platform)/build/stores/blockMenuStore";
import { cn } from "@/lib/utils";
import { blockMenuContainerStyle } from "../style";
import { NoSearchResult } from "../NoSearchResult";

export const BlockMenuSearchContent = () => {
  const {
    searchResults,
    isFetchingNextPage,
    fetchNextPage,
    hasNextPage,
    searchLoading,
    handleAddLibraryAgent,
    handleAddMarketplaceAgent,
    addingLibraryAgentId,
    addingMarketplaceAgentSlug,
  } = useBlockMenuSearchContent();

  const { searchQuery } = useBlockMenuStore();

  if (searchLoading) {
    return (
      <div
        className={cn(
          blockMenuContainerStyle,
          "flex items-center justify-center",
        )}
      >
        <LoadingSpinner className="size-13" />
      </div>
    );
  }

  if (searchResults.length === 0) {
    return <NoSearchResult />;
  }

  return (
    <InfiniteScroll
      isFetchingNextPage={isFetchingNextPage}
      fetchNextPage={fetchNextPage}
      hasNextPage={hasNextPage}
      loader={<LoadingSpinner className="size-13" />}
      className="space-y-2.5"
    >
      {searchResults.map((item: SearchResponseItemsItem, index: number) => {
        const { type, data } = getSearchItemType(item);
        // backend give support to these 3 types only [right now] - we need to give support to integration and ai agent types in follow up PRs
        switch (type) {
          case "store_agent":
            return (
              <MarketplaceAgentBlock
                key={index}
                slug={data.slug}
                highlightedText={searchQuery}
                title={data.agent_name}
                image_url={data.agent_image}
                creator_name={data.creator}
                number_of_runs={data.runs}
                loading={addingMarketplaceAgentSlug === data.slug}
                onClick={() =>
                  handleAddMarketplaceAgent({
                    creator_name: data.creator,
                    slug: data.slug,
                  })
                }
              />
            );
          case "block":
            return (
              <Block
                key={index}
                title={data.name}
                highlightedText={searchQuery}
                description={data.description}
                blockData={data}
              />
            );

          case "library_agent":
            return (
              <UGCAgentBlock
                key={index}
                title={data.name}
                highlightedText={searchQuery}
                image_url={data.image_url}
                version={data.graph_version}
                edited_time={data.updated_at}
                isLoading={addingLibraryAgentId === data.id}
                onClick={() => handleAddLibraryAgent(data)}
              />
            );

          default:
            return null;
        }
      })}
    </InfiniteScroll>
  );
};
