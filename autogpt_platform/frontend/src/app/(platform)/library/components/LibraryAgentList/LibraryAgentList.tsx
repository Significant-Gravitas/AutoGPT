"use client";
import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { LibraryActionSubHeader } from "../LibraryActionSubHeader/LibraryActionSubHeader";
import { LibraryAgentCard } from "../LibraryAgentCard/LibraryAgentCard";
import { useLibraryAgentList } from "./useLibraryAgentList";

interface Props {
  searchTerm: string;
  librarySort: LibraryAgentSort;
  setLibrarySort: (value: LibraryAgentSort) => void;
}

export function LibraryAgentList({
  searchTerm,
  librarySort,
  setLibrarySort,
}: Props) {
  const {
    agentLoading,
    agentCount,
    allAgents: agents,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
  } = useLibraryAgentList({ searchTerm, librarySort });

  return (
    <>
      <LibraryActionSubHeader
        agentCount={agentCount}
        setLibrarySort={setLibrarySort}
      />
      <div className="px-2">
        {agentLoading ? (
          <div className="flex h-[200px] items-center justify-center">
            <LoadingSpinner size="large" />
          </div>
        ) : (
          <InfiniteScroll
            isFetchingNextPage={isFetchingNextPage}
            fetchNextPage={fetchNextPage}
            hasNextPage={hasNextPage}
            loader={<LoadingSpinner size="medium" />}
          >
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {agents.map((agent) => (
                <LibraryAgentCard key={agent.id} agent={agent} />
              ))}
            </div>
          </InfiniteScroll>
        )}
      </div>
    </>
  );
}
