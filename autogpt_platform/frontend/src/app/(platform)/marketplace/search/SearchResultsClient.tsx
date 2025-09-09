"use client";

import { useState, useCallback } from "react";
import { AgentsSection } from "@/components/agptui/composite/AgentsSection";
import { FeaturedCreators } from "@/components/agptui/composite/FeaturedCreators";
import { Separator } from "@/components/ui/separator";
import { SearchFilterChips } from "@/components/agptui/SearchFilterChips";
import { SortDropdown } from "@/components/agptui/SortDropdown";
import { Creator, StoreAgent } from "@/lib/autogpt-server-api";
import { useRouter, useSearchParams } from "next/navigation";

export function SearchResultsClient({
  initialAgents,
  initialCreators,
  agentsCount,
  creatorsCount,
  totalCount,
}: {
  initialAgents: StoreAgent[];
  initialCreators: Creator[];
  agentsCount: number;
  creatorsCount: number;
  totalCount: number;
}) {
  const [showAgents, setShowAgents] = useState(true);
  const [showCreators, setShowCreators] = useState(true);
  const [agents, setAgents] = useState(initialAgents);
  const [creators, setCreators] = useState(initialCreators);
  const router = useRouter();
  const searchParams = useSearchParams();

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
      // Update URL with new sort parameter to trigger server-side re-fetch
      const params = new URLSearchParams(searchParams.toString());
      params.set("sort", sortValue);
      router.push(`/marketplace/search?${params.toString()}`);

      // Client-side sorting for immediate feedback
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
    [agents, creators, router, searchParams],
  );

  return (
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

        {showAgents && agentsCount > 0 && creatorsCount > 0 && showCreators && (
          <Separator />
        )}
        {showCreators && creatorsCount > 0 && (
          <FeaturedCreators featuredCreators={creators} title="Creators" />
        )}
      </div>
    </>
  );
}
