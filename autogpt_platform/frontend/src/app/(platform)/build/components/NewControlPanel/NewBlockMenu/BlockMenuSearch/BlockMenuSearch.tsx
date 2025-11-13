import { Text } from "@/components/atoms/Text/Text";
import { useBlockMenuSearch } from "./useBlockMenuSearch";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { LoadingSpinner } from "@/components/__legacy__/ui/loading";
import { SearchResponseItemsItem } from "@/app/api/__generated__/models/searchResponseItemsItem";
import { MarketplaceAgentBlock } from "../MarketplaceAgentBlock";
import { Block } from "../Block";
import { UGCAgentBlock } from "../UGCAgentBlock";
import { getSearchItemType } from "./helper";
import { useBlockMenuStore } from "../../../../stores/blockMenuStore";
import { blockMenuContainerStyle } from "../style";
import { cn } from "@/lib/utils";
import { NoSearchResult } from "../NoSearchResult";
import { useNodeStore } from "../../../../stores/nodeStore";

export const BlockMenuSearch = () => {
  const {
    allSearchData,
    isFetchingNextPage,
    fetchNextPage,
    hasNextPage,
    searchLoading,
  } = useBlockMenuSearch();
  const { searchQuery } = useBlockMenuStore();
  const addBlock = useNodeStore((state) => state.addBlock);

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

  if (allSearchData.length === 0) {
    return <NoSearchResult />;
  }

  return (
    <div className={blockMenuContainerStyle}>
      <Text variant="body-medium">Search results</Text>
      <InfiniteScroll
        isFetchingNextPage={isFetchingNextPage}
        fetchNextPage={fetchNextPage}
        hasNextPage={hasNextPage}
        loader={<LoadingSpinner className="size-13" />}
        className="space-y-2.5"
      >
        {allSearchData.map((item: SearchResponseItemsItem, index: number) => {
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
                  loading={false}
                />
              );
            case "block":
              return (
                <Block
                  key={index}
                  title={data.name}
                  highlightedText={searchQuery}
                  description={data.description}
                  onClick={() => addBlock(data)}
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
                />
              );

            default:
              return null;
          }
        })}
      </InfiniteScroll>
    </div>
  );
};
