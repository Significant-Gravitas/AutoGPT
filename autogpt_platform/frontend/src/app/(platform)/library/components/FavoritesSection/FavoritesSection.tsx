"use client";

import React from "react";
import { useFavoriteAgents } from "../../hooks/useFavoriteAgents";
import LibraryAgentCard from "../LibraryAgentCard/LibraryAgentCard";
import { useGetFlag, Flag } from "@/services/feature-flags/use-get-flag";
import { Heart } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

export default function FavoritesSection() {
  const isAgentFavoritingEnabled = useGetFlag(Flag.AGENT_FAVORITING);
  const {
    allAgents: favoriteAgents,
    agentLoading: isLoading,
    agentCount,
    hasNextPage,
    fetchNextPage,
    isFetchingNextPage,
  } = useFavoriteAgents();

  // Only show this section if the feature flag is enabled
  if (!isAgentFavoritingEnabled) {
    return null;
  }

  // Don't show the section if there are no favorites
  if (!isLoading && favoriteAgents.length === 0) {
    return null;
  }

  return (
    <div className="mb-8">
      <div className="flex items-center gap-[10px] p-2 pb-[10px]">
        <Heart className="h-5 w-5 fill-red-500 text-red-500" />
        <span className="font-poppin text-[18px] font-semibold leading-[28px] text-neutral-800">
          Favorites
        </span>
        {!isLoading && (
          <span className="font-sans text-[14px] font-normal leading-6">
            {agentCount} {agentCount === 1 ? "agent" : "agents"}
          </span>
        )}
      </div>

      <div className="relative">
        {isLoading ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} className="h-48 w-full rounded-lg" />
            ))}
          </div>
        ) : (
          <InfiniteScroll
            isFetchingNextPage={isFetchingNextPage}
            fetchNextPage={fetchNextPage}
            hasNextPage={hasNextPage}
            loader={
              <div className="flex h-8 w-full items-center justify-center">
                <div className="h-6 w-6 animate-spin rounded-full border-b-2 border-t-2 border-neutral-800" />
              </div>
            }
          >
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {favoriteAgents.map((agent: LibraryAgent) => (
                <LibraryAgentCard key={agent.id} agent={agent} />
              ))}
            </div>
          </InfiniteScroll>
        )}
      </div>

      {favoriteAgents.length > 0 && <div className="mt-6 border-t pt-6" />}
    </div>
  );
}
