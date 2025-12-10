import React, { useEffect, useRef, useState } from "react";
import { IntegrationChip } from "../IntegrationChip";
import { Block } from "../Block";
import { useSuggestionContent } from "./useSuggestionContent";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { blockMenuContainerStyle } from "../style";
import { useBlockMenuStore } from "../../../../stores/blockMenuStore";
import { DefaultStateType } from "../types";
import { SearchHistoryChip } from "../SearchHistoryChip";
import { ArrowLeftIcon, ArrowRightIcon } from "@phosphor-icons/react";

export const SuggestionContent = () => {
  const { setIntegration, setDefaultState, setSearchQuery, setSearchId } =
    useBlockMenuStore();
  const recentSearchesRef = useRef<HTMLDivElement | null>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  const scrollRecentSearches = (delta: number) => {
    if (!recentSearchesRef.current) {
      return;
    }
    recentSearchesRef.current.scrollBy({ left: delta, behavior: "smooth" });
  };
  const updateScrollState = () => {
    const element = recentSearchesRef.current;
    if (!element) {
      setCanScrollLeft(false);
      setCanScrollRight(false);
      return;
    }
    setCanScrollLeft(element.scrollLeft > 0);
    setCanScrollRight(
      Math.ceil(element.scrollLeft + element.clientWidth) < element.scrollWidth,
    );
  };
  const { data, isLoading, isError, error, refetch } = useSuggestionContent();
  const suggestions = data?.suggestions;

  useEffect(() => {
    updateScrollState();
    const element = recentSearchesRef.current;
    if (!element) {
      return;
    }
    const handleScroll = () => updateScrollState();
    element.addEventListener("scroll", handleScroll);
    window.addEventListener("resize", handleScroll);
    return () => {
      element.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", handleScroll);
    };
  }, [suggestions?.recent_searches?.length, isLoading]);

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
        <div className="space-y-2.5 px-4">
          <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
            Recent searches
          </p>
          <div className="-mx-8">
            <div className="group relative">
              <div
                ref={recentSearchesRef}
                className="flex gap-2 overflow-x-auto px-8 [scrollbar-width:none] [-ms-overflow-style:'none'] [&::-webkit-scrollbar]:hidden"
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
              </div>
              {canScrollLeft && (
                <div className="pointer-events-none absolute inset-y-0 left-0 w-8 bg-gradient-to-r from-white via-white/80 to-white/0" />
              )}
              {canScrollRight && (
                <div className="pointer-events-none absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-white via-white/80 to-white/0" />
              )}
              {canScrollLeft && (
                <button
                  type="button"
                  aria-label="Scroll recent searches left"
                  className="pointer-events-none absolute left-2 top-5 -translate-y-1/2 opacity-0 transition-opacity duration-200 group-hover:pointer-events-auto group-hover:opacity-100"
                  onClick={() => scrollRecentSearches(-300)}
                >
                  <ArrowLeftIcon
                    size={28}
                    className="text-white bg-zinc-700 drop-shadow rounded-full p-1"
                    weight="light"
                  />
                </button>
              )}
              {canScrollRight && (
                <button
                  type="button"
                  aria-label="Scroll recent searches right"
                  className="pointer-events-none absolute right-2 top-5 -translate-y-1/2 opacity-0 transition-opacity duration-200 group-hover:pointer-events-auto group-hover:opacity-100"
                  onClick={() => scrollRecentSearches(300)}
                >
                  <ArrowRightIcon
                    size={28}
                    className="text-white bg-zinc-700 drop-shadow rounded-full p-1"
                    weight="light"
                  />
                </button>
              )}
            </div>
          </div>
        </div>
        
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
