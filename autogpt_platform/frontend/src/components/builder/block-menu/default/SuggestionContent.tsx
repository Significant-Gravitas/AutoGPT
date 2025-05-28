import React, { useEffect, useState } from "react";
import SearchHistoryChip from "../SearchHistoryChip";
import IntegrationChip from "../IntegrationChip";
import Block from "../Block";
import { useBlockMenuContext } from "../block-menu-provider";
import {
  CredentialsProviderName,
  SuggestionsResponse,
} from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

const SuggestionContent: React.FC = () => {
  const { setIntegration, setDefaultState, setSearchQuery, addNode } =
    useBlockMenuContext();

  const [suggestionsData, setSuggestionsData] =
    useState<SuggestionsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const api = useBackendAPI();

  useEffect(() => {
    const fetchSuggestions = async () => {
      try {
        setLoading(true);
        const response = await api.getSuggestions();
        setSuggestionsData(response);
        console.log(response);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchSuggestions();
  }, [api]);

  return (
    <div className="scrollbar-thumb-rounded h-full overflow-y-auto pt-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-transparent hover:scrollbar-thumb-zinc-200 transition-all duration-200">
      <div className="w-full space-y-6 pb-4">
        {/* Recent Searches */}
        <div className="space-y-2.5">
          <p className="px-4 font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
            Recent searches
          </p>
          <div className="flex flex-nowrap gap-2 overflow-x-auto scrollbar-hide">
            {!loading && suggestionsData
              ? suggestionsData.recent_searches.map((search, index) => (
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
            {!loading && suggestionsData
              ? suggestionsData.providers.map((provider, index) => (
                  <IntegrationChip
                    key={`integration-${index}`}
                    icon_url={`/integrations/${provider}.png`}
                    name={provider}
                    onClick={() => {
                      setDefaultState("integrations");
                      setIntegration(provider as CredentialsProviderName);
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
            {!loading && suggestionsData
              ? suggestionsData.top_blocks.map((block, index) => (
                  <Block
                    key={`block-${index}`}
                    title={block.name}
                    description={block.description}
                    onClick={() => {
                      addNode(
                        block.id,
                        block.name,
                        block.hardcodedValues || {},
                        block,
                      );
                    }}
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
