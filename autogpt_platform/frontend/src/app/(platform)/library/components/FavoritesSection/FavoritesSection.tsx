"use client";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import { Text } from "@/components/atoms/Text/Text";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { HeartIcon } from "@phosphor-icons/react";
import { useFavoriteAgents } from "../../hooks/useFavoriteAgents";
import { LibraryAgentCard } from "../LibraryAgentCard/LibraryAgentCard";
import { LibraryTabs, Tab } from "../LibraryTabs/LibraryTabs";
import { LibraryActionSubHeader } from "../LibraryActionSubHeader/LibraryActionSubHeader";

interface Props {
  searchTerm: string;
  tabs: Tab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
  setLibrarySort: (value: LibraryAgentSort) => void;
}

export function FavoritesSection({
  searchTerm,
  tabs,
  activeTab,
  onTabChange,
  setLibrarySort,
}: Props) {
  const {
    allAgents: favoriteAgents,
    agentLoading: isLoading,
    agentCount,
    hasNextPage,
    fetchNextPage,
    isFetchingNextPage,
  } = useFavoriteAgents({ searchTerm });

  return (
    <>
      <LibraryActionSubHeader
        agentCount={agentCount}
        setLibrarySort={setLibrarySort}
      />
      <LibraryTabs
        tabs={tabs}
        activeTab={activeTab}
        onTabChange={onTabChange}
      />

      {isLoading ? (
        <div className="flex h-[200px] items-center justify-center">
          <LoadingSpinner size="large" />
        </div>
      ) : favoriteAgents.length === 0 ? (
        <div className="flex h-[200px] flex-col items-center justify-center gap-2 text-zinc-500">
          <HeartIcon className="h-10 w-10" />
          <Text variant="body">No favorite agents yet</Text>
        </div>
      ) : (
        <InfiniteScroll
          isFetchingNextPage={isFetchingNextPage}
          fetchNextPage={fetchNextPage}
          hasNextPage={hasNextPage}
          loader={<LoadingSpinner size="medium" />}
        >
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {favoriteAgents.map((agent: LibraryAgent) => (
              <LibraryAgentCard key={agent.id} agent={agent} />
            ))}
          </div>
        </InfiniteScroll>
      )}
    </>
  );
}
