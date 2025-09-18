"use client";

import { use, useCallback, useEffect, useState } from "react";
import { AgentsSection } from "@/components/__legacy__/composite/AgentsSection";
import { SearchBar } from "@/components/__legacy__/SearchBar";
import { FeaturedCreators } from "@/components/__legacy__/composite/FeaturedCreators";
import { Separator } from "@/components/__legacy__/ui/separator";
import { SearchFilterChips } from "@/components/__legacy__/SearchFilterChips";
import { SortDropdown } from "@/components/__legacy__/SortDropdown";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { Creator, StoreAgent } from "@/lib/autogpt-server-api";

type MarketplaceSearchPageSearchParams = { searchTerm?: string; sort?: string };

export default function MarketplaceSearchPage({
  searchParams,
}: {
  searchParams: Promise<MarketplaceSearchPageSearchParams>;
}) {
  return (
    <SearchResults
      searchTerm={use(searchParams).searchTerm || ""}
      sort={use(searchParams).sort || "trending"}
    />
  );
}

function SearchResults({
  searchTerm,
  sort,
}: {
  searchTerm: string;
  sort: string;
}): React.ReactElement {
  const [showAgents, setShowAgents] = useState(true);
  const [showCreators, setShowCreators] = useState(true);
  const [agents, setAgents] = useState<StoreAgent[]>([]);
  const [creators, setCreators] = useState<Creator[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const api = useBackendAPI();

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);

      try {
        const [agentsRes, creatorsRes] = await Promise.all([
          api.getStoreAgents({
            search_query: searchTerm,
            sorted_by: sort,
          }),
          api.getStoreCreators({
            search_query: searchTerm,
          }),
        ]);

        setAgents(agentsRes.agents || []);
        setCreators(creatorsRes.creators || []);
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [api, searchTerm, sort]);

  const agentsCount = agents.length;
  const creatorsCount = creators.length;
  const totalCount = agentsCount + creatorsCount;

  const handleFilterChange = (value: string) => {
    if (value === "agents") {
      setShowAgents(true);
      setShowCreators(false);
    } else if (value === "creators") {
      setShowAgents(false);
      setShowCreators(true);
    } else {
      setShowAgents(true);
      setShowCreators(true);
    }
  };

  const handleSortChange = useCallback(
    (sortValue: string) => {
      let sortBy = "recent";
      if (sortValue === "runs") {
        sortBy = "runs";
      } else if (sortValue === "rating") {
        sortBy = "rating";
      }

      const sortedAgents = [...agents].sort((a, b) => {
        if (sortBy === "runs") {
          return b.runs - a.runs;
        } else if (sortBy === "rating") {
          return b.rating - a.rating;
        } else {
          return (
            new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
          );
        }
      });

      const sortedCreators = [...creators].sort((a, b) => {
        if (sortBy === "runs") {
          return b.agent_runs - a.agent_runs;
        } else if (sortBy === "rating") {
          return b.agent_rating - a.agent_rating;
        } else {
          // Creators don't have updated_at, sort by number of agents as fallback
          return b.num_agents - a.num_agents;
        }
      });

      setAgents(sortedAgents);
      setCreators(sortedCreators);
    },
    [agents, creators],
  );

  return (
    <div className="w-full">
      <div className="mx-auto min-h-screen max-w-[1440px] px-10 lg:min-w-[1440px]">
        <div className="mt-8 flex items-center">
          <div className="flex-1">
            <h2 className="text-base font-medium leading-normal text-neutral-800 dark:text-neutral-200">
              Results for:
            </h2>
            <h1 className="font-poppins text-2xl font-semibold leading-[32px] text-neutral-800 dark:text-neutral-100">
              {searchTerm}
            </h1>
          </div>
          <div className="flex-none">
            <SearchBar width="w-[439px]" height="h-[60px]" />
          </div>
        </div>

        {isLoading ? (
          <div className="mt-20 flex flex-col items-center justify-center">
            <p className="text-neutral-500 dark:text-neutral-400">Loading...</p>
          </div>
        ) : totalCount > 0 ? (
          <>
            <div className="mt-[36px] flex items-center justify-between">
              <SearchFilterChips
                totalCount={totalCount}
                agentsCount={agentsCount}
                creatorsCount={creatorsCount}
                onFilterChange={handleFilterChange}
              />
              <SortDropdown onSort={handleSortChange} />
            </div>
            {/* Content section */}
            <div className="min-h-[500px] max-w-[1440px]">
              {showAgents && agentsCount > 0 && (
                <div className="mt-[36px]">
                  <AgentsSection agents={agents} sectionTitle="Agents" />
                </div>
              )}

              {showAgents && agentsCount > 0 && creatorsCount > 0 && (
                <Separator />
              )}
              {showCreators && creatorsCount > 0 && (
                <FeaturedCreators
                  featuredCreators={creators}
                  title="Creators"
                />
              )}
            </div>
          </>
        ) : (
          <div className="mt-20 flex flex-col items-center justify-center">
            <h3 className="mb-2 text-xl font-medium text-neutral-600 dark:text-neutral-300">
              No results found
            </h3>
            <p className="text-neutral-500 dark:text-neutral-400">
              Try adjusting your search terms or filters
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
