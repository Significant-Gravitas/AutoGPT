"use client";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { HeartIcon } from "@phosphor-icons/react";
import { useFavoriteAgents } from "../../hooks/useFavoriteAgents";
import { LibraryAgentCard } from "../LibraryAgentCard/LibraryAgentCard";

interface Props {
  searchTerm: string;
}

export function FavoritesSection({ searchTerm }: Props) {
  const {
    allAgents: favoriteAgents,
    agentLoading: isLoading,
    agentCount,
    hasNextPage,
    fetchNextPage,
    isFetchingNextPage,
  } = useFavoriteAgents({ searchTerm });

  if (isLoading || favoriteAgents.length === 0) {
    return null;
  }

  return (
    <div className="!mb-8">
      <div className="mb-3 flex items-center gap-2 p-2">
        <HeartIcon className="h-5 w-5" weight="fill" />
        <div className="flex items-baseline gap-2">
          <Text variant="h4">Favorites</Text>
          {!isLoading && (
            <Text
              variant="body"
              data-testid="agents-count"
              className="relative bottom-px text-zinc-500"
            >
              {agentCount}
            </Text>
          )}
        </div>
      </div>

      <div className="relative">
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
      </div>

      {favoriteAgents.length > 0 && <div className="!mt-10 border-t" />}
    </div>
  );
}
