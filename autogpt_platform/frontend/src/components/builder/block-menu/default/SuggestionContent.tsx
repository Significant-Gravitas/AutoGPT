import React, { useEffect, useState } from "react";
import SearchHistoryChip from "../SearchHistoryChip";
import IntegrationChip from "../IntegrationChip";
import Block from "../Block";
import { DefaultStateType } from "./BlockMenuDefault";
import {
  integrationsData,
  topBlocksData,
  recentSearchesData,
} from "../../testing_data";

interface SuggestionContentProps {
  setIntegration: React.Dispatch<React.SetStateAction<string>>;
  setDefaultState: React.Dispatch<React.SetStateAction<DefaultStateType>>;
  setSearchQuery: React.Dispatch<React.SetStateAction<string>>;
}

const SuggestionContent: React.FC<SuggestionContentProps> = ({
  setIntegration,
  setDefaultState,
  setSearchQuery,
}) => {
  const [recentSearches, setRecentSearches] = useState<string[] | null>(null);
  const [integrations, setIntegrations] = useState<
    { icon_url: string; name: string }[] | null
  >(null);
  const [topBlocks, setTopBlocks] = useState<
    { title: string; description: string }[] | null
  >(null);

  // TEMPORARY FETCHING
  useEffect(() => {
    const fetchData = async () => {
      try {
        const fetchRecentSearches = async (): Promise<string[]> => {
          await new Promise((resolve) => setTimeout(resolve, 300));
          return recentSearchesData;
        };

        const fetchIntegrations = async (): Promise<
          { icon_url: string; name: string }[]
        > => {
          await new Promise((resolve) => setTimeout(resolve, 400));
          return integrationsData;
        };

        const fetchTopBlocks = async (): Promise<
          { title: string; description: string }[]
        > => {
          await new Promise((resolve) => setTimeout(resolve, 600));
          return topBlocksData;
        };

        const [
          recentSearchesDataFetched,
          integrationsDataFetched,
          topBlocksDataFetched,
        ] = await Promise.all([
          fetchRecentSearches(),
          fetchIntegrations(),
          fetchTopBlocks(),
        ]);

        setRecentSearches(recentSearchesDataFetched);
        setIntegrations(integrationsDataFetched);
        setTopBlocks(topBlocksDataFetched);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="scrollbar-thumb-rounded h-full overflow-y-scroll pt-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200">
      <div className="w-full space-y-6 pb-4">
        {/* Recent Searches */}
        <div className="space-y-2.5">
          <p className="px-4 font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
            Recent searches
          </p>
          <div className="flex flex-nowrap gap-2 overflow-x-auto scrollbar-hide">
            {recentSearches
              ? recentSearches.map((search, index) => (
                  <SearchHistoryChip
                    key={`search-${index}`}
                    content={search}
                    className={index === 0 ? "ml-4" : ""}
                    onClick={() => setSearchQuery(search)}
                  />
                ))
              : Array(3)
                  .fill(0)
                  .map((_, index) => (
                    <SearchHistoryChip.Skeleton
                      key={`search-${index}`}
                      className={index === 0 ? "ml-4" : ""}
                    />
                  ))}
          </div>
        </div>

        {/* Integrations */}
        <div className="space-y-2.5 px-4">
          <p className="font-sans text-xs font-medium leading-[1.25rem] text-zinc-500">
            Integrations
          </p>
          <div className="grid grid-cols-3 grid-rows-2 gap-2">
            {integrations
              ? integrations.map((integration, index) => (
                  <IntegrationChip
                    key={`integration-${index}`}
                    icon_url={integration.icon_url}
                    name={integration.name}
                    onClick={() => {
                      setDefaultState("integrations");
                      setIntegration(integration.name);
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
            {topBlocks
              ? topBlocks.map((block, index) => (
                  <Block
                    key={`block-${index}`}
                    title={block.title}
                    description={block.description}
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

export default SuggestionContent;
