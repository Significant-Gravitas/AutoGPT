"use client";

import { useEffect, useRef } from "react";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import { Text } from "@/components/atoms/Text/Text";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import {
  TabsLine,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { HeartIcon, Icon } from "@phosphor-icons/react";
import { useFavoriteAnimation } from "../../context/FavoriteAnimationContext";
import { useFavoriteAgents } from "../../hooks/useFavoriteAgents";
import { LibraryAgentCard } from "../LibraryAgentCard/LibraryAgentCard";
import { LibraryActionSubHeader } from "../LibraryActionSubHeader/LibraryActionSubHeader";

interface Props {
  searchTerm: string;
  tabs: { id: string; title: string; icon: Icon }[];
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

  const { registerFavoritesTabRef } = useFavoriteAnimation();
  const favoritesRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    registerFavoritesTabRef(favoritesRef.current);
    return () => {
      registerFavoritesTabRef(null);
    };
  }, [registerFavoritesTabRef]);

  return (
    <>
      <LibraryActionSubHeader
        agentCount={agentCount}
        setLibrarySort={setLibrarySort}
      />
      <TabsLine value={activeTab} onValueChange={onTabChange}>
        <TabsLineList>
          {tabs.map((tab) => (
            <TabsLineTrigger
              key={tab.id}
              value={tab.id}
              ref={tab.id === "favorites" ? favoritesRef : undefined}
              className="inline-flex items-center gap-1.5"
            >
              <tab.icon size={16} />
              {tab.title}
            </TabsLineTrigger>
          ))}
        </TabsLineList>
      </TabsLine>

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
