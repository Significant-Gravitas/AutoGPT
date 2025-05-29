import React from "react";
import MarketplaceAgentBlock from "../MarketplaceAgentBlock";
import Block from "../Block";
import UGCAgentBlock from "../UGCAgentBlock";
import AiBlock from "./AiBlock";
import IntegrationBlock from "../IntegrationBlock";
import { SearchItem, useBlockMenuContext } from "../block-menu-provider";
import NoSearchResult from "./NoSearchResult";
import { Button } from "@/components/ui/button";

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
  const { searchQuery, addNode, searchData } = useBlockMenuContext();

  // Need to change it once, we got provider blocks
  const getBlockType = (item: any) => {
    if (item.id && item.name && item.inputSchema && item.outputSchema) {
      return "block";
    }
    if (item.name && typeof item.integration_count === "number") {
      return "provider";
    }
    if (item.id && item.graph_id && item.status) {
      return "library_agent";
    }
    if (item.slug && item.agent_name && item.runs !== undefined) {
      return "store_agent";
    }
    return null;
  };

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
                highlightedText={searchQuery}
                title={item.agent_name}
                image_url={item.agent_image}
                creator_name={item.creator}
                number_of_runs={item.runs}
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
            // Here we do need the Integration blocks list, not integration itself
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
              />
            );
          // currently our backend does not support ai blocks
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
