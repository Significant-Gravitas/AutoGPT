import React from "react";
import { IntegrationChip } from "../IntegrationChip";
import { Block } from "../Block";
import { useSuggestionContent } from "./useSuggestionContent";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { blockMenuContainerStyle } from "../style";
import { useBlockMenuStore } from "../../../../stores/blockMenuStore";
import { DefaultStateType } from "../types";
import { SearchHistoryChip } from "../SearchHistoryChip";
import { HorizontalScroll } from "../HorizontalScroll";

export const SuggestionContent = () => {
  const { setIntegration, setDefaultState, setSearchQuery, setSearchId } =
    useBlockMenuStore();
  const { data, isLoading, isError, error, refetch } = useSuggestionContent();
  const suggestions = data?.suggestions;
  const hasRecentSearches = (suggestions?.recent_searches?.length ?? 0) > 0;

  if (isError) {
    return (
      <div className="h-full p-4">
        <ErrorCard
          isSuccess={false}
          responseError={error || undefined}
          httpError={{
            status: data?.status,
            statusText: "Request failed",
            message: (error?.detail as string) || "An error occurred",
          }}
          context="block menu"
          onRetry={() => refetch()}
        />
      </div>
    );
  }

  return (
    <div className={blockMenuContainerStyle}>
      <div className="w-full space-y-6 pb-4">
        {/* Recent searches */}
        {hasRecentSearches && (
          <div className="space-y-2.5 px-4">
            <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
              Recent searches
            </p>
            <HorizontalScroll
              wrapperClassName="-mx-8"
              scrollContainerClassName="flex gap-2 overflow-x-auto px-8 [scrollbar-width:none] [-ms-overflow-style:'none'] [&::-webkit-scrollbar]:hidden"
              dependencyList={[
                suggestions?.recent_searches?.length ?? 0,
                isLoading,
              ]}
            >
              {!isLoading && suggestions
                ? suggestions.recent_searches.map((entry, index) => (
                    <SearchHistoryChip
                      key={entry.search_id || `${entry.search_query}-${index}`}
                      content={entry.search_query || "Untitled search"}
                      onClick={() => {
                        setSearchQuery(entry.search_query || "");
                        setSearchId(entry.search_id || undefined);
                      }}
                    />
                  ))
                : Array(3)
                    .fill(0)
                    .map((_, index) => (
                      <SearchHistoryChip.Skeleton
                        key={`recent-search-skeleton-${index}`}
                      />
                    ))}
            </HorizontalScroll>
          </div>
        )}

        {/* Integrations */}
        <div className="space-y-2.5 px-4">
          <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
            Integrations
          </p>
          <div className="grid grid-cols-3 grid-rows-2 gap-2">
            {!isLoading && suggestions
              ? suggestions.providers.map((provider, index) => (
                  <IntegrationChip
                    key={`integration-${index}`}
                    icon_url={`/integrations/${provider}.png`}
                    name={provider}
                    onClick={() => {
                      setDefaultState(DefaultStateType.INTEGRATIONS);
                      setIntegration(provider);
                    }}
                  />
                ))
              : Array(6)
                  .fill(0)
                  .map((_, index) => (
                    <IntegrationChip.Skeleton
                      key={`integration-skeleton-${index}`}
                    />
                  ))}
          </div>
        </div>

        {/* Top blocks */}
        <div className="space-y-2.5 px-4">
          <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
            Top blocks
          </p>
          <div className="space-y-2">
            {!isLoading && suggestions
              ? suggestions.top_blocks.map((block, index) => (
                  <Block
                    key={`block-${index}`}
                    title={block.name}
                    description={block.description}
                    blockData={block}
                  />
                ))
              : Array(3)
                  .fill(0)
                  .map((_, index) => (
                    <Block.Skeleton key={`block-skeleton-${index}`} />
                  ))}
          </div>
        </div>
      </div>
    </div>
  );
};
