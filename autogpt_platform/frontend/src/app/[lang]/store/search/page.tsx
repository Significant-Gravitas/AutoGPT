"use client";

import { useState, useEffect } from "react";
import AutoGPTServerAPIClient from "@/lib/autogpt-server-api/client";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { SearchBar } from "@/components/agptui/SearchBar";
import { FeaturedCreators } from "@/components/agptui/composite/FeaturedCreators";
import { Separator } from "@/components/ui/separator";
import { SearchFilterChips } from "@/components/agptui/SearchFilterChips";
import { SortDropdown } from "@/components/agptui/SortDropdown";

export default function Page({
  params,
  searchParams,
}: {
  params: { lang: string };
  searchParams: { searchTerm?: string; sort?: string };
}) {
  return (
    <SearchResults 
      searchTerm={searchParams.searchTerm || ""}
      sort={searchParams.sort || "trending"}
    />
  );
}

function SearchResults({
  searchTerm,
  sort
}: {
  searchTerm: string;
  sort: string;
}) {
  const [showAgents, setShowAgents] = useState(true);
  const [showCreators, setShowCreators] = useState(true);
  const [agents, setAgents] = useState<any[]>([]);
  const [creators, setCreators] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      const api = new AutoGPTServerAPIClient();
      
      try {
        const [agentsRes, creatorsRes] = await Promise.all([
          api.getStoreAgents({ 
            search_query: searchTerm,
            sorted_by: sort 
          }),
          api.getStoreCreators({
            search_query: searchTerm,
          })
        ]);

        setAgents(agentsRes.agents || []);
        setCreators(creatorsRes.creators || []);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [searchTerm, sort]);

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

  return (
    <div className="w-full">
      <div className="px-10 max-w-[1440px] lg:min-w-[1440px] mx-auto min-h-screen">
        <div className="flex items-center mt-8">
          <div className="flex-1">
            <h2 className="font-['Poppins'] text-base font-medium text-neutral-800 dark:text-neutral-200">
              Results for:
            </h2>
            <h1 className="font-['Poppins'] text-2xl font-semibold text-neutral-800 dark:text-neutral-100">
              {searchTerm}
            </h1>
          </div>
          <div className="flex-none">
            <SearchBar width="w-[439px]" />
          </div>
        </div>

        {isLoading ? (
          <div className="flex flex-col items-center justify-center mt-20">
            <p className="text-neutral-500 dark:text-neutral-400">Loading...</p>
          </div>
        ) : totalCount > 0 ? (
          <>
            <div className="mt-8 flex justify-between items-center">
              <SearchFilterChips 
                totalCount={totalCount}
                agentsCount={agentsCount}
                creatorsCount={creatorsCount}
                onFilterChange={handleFilterChange}
              />
              <SortDropdown />
            </div>
            {/* Content section */}
            <div className="max-w-[1440px] min-h-[500px]">
              {showAgents && agentsCount > 0 && (
                <div className="mt-8">
                  <AgentsSection agents={agents} sectionTitle="Agents" />
                </div>
              )}
              
              {showAgents && agentsCount > 0 && creatorsCount > 0 && <Separator />}
              {showCreators && creatorsCount > 0 && (
                <FeaturedCreators featuredCreators={creators} title="Creators" />
              )}
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center mt-20">
            <h3 className="text-xl font-medium text-neutral-600 dark:text-neutral-300 mb-2">No results found</h3>
            <p className="text-neutral-500 dark:text-neutral-400">Try adjusting your search terms or filters</p>
          </div>
        )}
      </div>
    </div>
  );
}
