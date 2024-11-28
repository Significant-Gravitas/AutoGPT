"use client";

import { useState } from "react";
import AutoGPTServerAPIClient from "@/lib/autogpt-server-api/client";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { SearchBar } from "@/components/agptui/SearchBar";
import { FeaturedCreators } from "@/components/agptui/composite/FeaturedCreators";
import { Separator } from "@/components/ui/separator";
import { SearchFilterChips } from "@/components/agptui/SearchFilterChips";
import { SortDropdown } from "@/components/agptui/SortDropdown";

export default async function Page({
  params,
  searchParams,
}: {
  params: { lang: string };
  searchParams: { searchTerm?: string; sort?: string };
}) {
  const search_term = searchParams.searchTerm || "";
  const sort = searchParams.sort || "trending";
  
  const api = new AutoGPTServerAPIClient();
  const { agents } = await api.getStoreAgents({ 
    search_query: search_term,
    sorted_by: sort 
  });
  const { creators } = await api.getStoreCreators({
    search_query: search_term,
  });

  const agentsCount = agents?.length || 0;
  const creatorsCount = creators?.length || 0;
  const totalCount = agentsCount + creatorsCount;

  // Move state to client component
  return (
    <SearchResults 
      search_term={search_term}
      agents={agents}
      creators={creators}
      agentsCount={agentsCount} 
      creatorsCount={creatorsCount}
      totalCount={totalCount}
    />
  );
}

function SearchResults({
  search_term,
  agents,
  creators,
  agentsCount,
  creatorsCount, 
  totalCount
}: {
  search_term: string;
  agents: any[];
  creators: any[];
  agentsCount: number;
  creatorsCount: number;
  totalCount: number;
}) {
  const [filter, setFilter] = useState("all");
  const [showAgents, setShowAgents] = useState(true);
  const [showCreators, setShowCreators] = useState(true);

  const handleFilterChange = (value: string) => {
    setFilter(value);
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
    <div className="w-full bg-white">
      <div className="px-10 max-w-[1440px] mx-auto min-h-screen">
        <div className="flex items-center mt-8">
          <div className="flex-1">
            <h2 className="font-['Poppins'] text-base font-medium text-neutral-800">
              Results for:
            </h2>
            <h1 className="font-['Poppins'] text-2xl font-semibold text-neutral-800">
              {search_term}
            </h1>
          </div>
          <div className="flex-none">
            <SearchBar width="w-[439px]" />
          </div>
        </div>

        {totalCount > 0 ? (
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
            <h3 className="text-xl font-medium text-neutral-600 mb-2">No results found</h3>
            <p className="text-neutral-500">Try adjusting your search terms or filters</p>
          </div>
        )}
      </div>
    </div>
  );
}
