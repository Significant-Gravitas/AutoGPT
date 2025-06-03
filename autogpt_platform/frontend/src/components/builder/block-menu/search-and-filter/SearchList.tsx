import React from "react";
import MarketplaceAgentBlock from "../MarketplaceAgentBlock";
import Block from "../Block";
import UGCAgentBlock from "../UGCAgentBlock";
import AiBlock from "./AiBlock";
import IntegrationBlock from "../IntegrationBlock";
import { SearchItem, useBlockMenuContext } from "../block-menu-provider";
import NoSearchResult from "./NoSearchResult";
import { Button } from "@/components/ui/button";
import { convertLibraryAgentIntoBlock, getBlockType } from "@/lib/utils";

interface SearchListProps {
  isLoading: boolean;
  loadingMore: boolean;
  hasMore: boolean;
  error: string | null;
  onRetry: () => void;
}

const SearchList: React.FC<SearchListProps> = ({
  isLoading,
  loadingMore,
  hasMore,
  error,
  onRetry,
}) => {
  const { searchQuery, addNode, loadingSlug, searchData, handleAddStoreAgent } =
    useBlockMenuContext();

  if (isLoading) {
    return (
      <div className="space-y-2.5 px-4">
        <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
          Search results
        </p>
        {Array(6)
          .fill(0)
          .map((_, i) => (
            <Block.Skeleton key={i} />
          ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="px-4">
        <div className="rounded-lg border border-red-200 bg-red-50 p-3">
          <p className="mb-2 text-sm text-red-600">
            Error loading search results: {error}
          </p>
          <Button
            variant="outline"
            size="sm"
            onClick={onRetry}
            className="h-7 text-xs"
          >
            Retry
          </Button>
        </div>
      </div>
    );
  }

  if (searchData.length === 0) {
    return <NoSearchResult />;
  }

  return (
    <div className="space-y-2.5 px-4">
      <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
        Search results
      </p>
      {searchData.map((item: any, index: number) => {
        const blockType = getBlockType(item);

        switch (blockType) {
          case "store_agent":
            return (
              <MarketplaceAgentBlock
                key={index}
                slug={item.slug}
                highlightedText={searchQuery}
                title={item.agent_name}
                image_url={item.agent_image}
                creator_name={item.creator}
                number_of_runs={item.runs}
                loading={loadingSlug == item.slug}
                onClick={() =>
                  handleAddStoreAgent({
                    creator_name: item.creator,
                    slug: item.slug,
                  })
                }
              />
            );
          case "block":
            return (
              <Block
                key={index}
                title={item.name}
                highlightedText={searchQuery}
                description={item.description}
                onClick={() => {
                  addNode(item);
                }}
              />
            );
          case "provider":
            return (
              <IntegrationBlock
                key={index}
                title={item.name}
                highlightedText={searchQuery}
                icon_url={`/integrations/${item.name}.png`}
                description={item.description}
                onClick={() => {
                  addNode(item);
                }}
              />
            );
          case "library_agent":
            return (
              <UGCAgentBlock
                key={index}
                title={item.name}
                highlightedText={searchQuery}
                image_url={item.image_url}
                version={item.graph_version}
                edited_time={item.updated_at}
                onClick={() => {
                  const block = convertLibraryAgentIntoBlock(item);
                  addNode(block);
                }}
              />
            );
          case "ai_agent":
            return (
              <AiBlock
                key={index}
                title={item.name}
                description={item.description}
                ai_name={item.inputSchema.properties.model.enum.find(
                  (model: string) =>
                    model
                      .toLowerCase()
                      .includes(searchQuery.toLowerCase().trim()),
                )}
                onClick={() => {
                  const block = convertLibraryAgentIntoBlock(item);
                  addNode(block);
                }}
              />
            );

          default:
            return null;
        }
      })}
      {loadingMore && hasMore && (
        <div className="space-y-2.5">
          {Array(3)
            .fill(0)
            .map((_, i) => (
              <Block.Skeleton key={`loading-more-${i}`} />
            ))}
        </div>
      )}
    </div>
  );
};

export default SearchList;
