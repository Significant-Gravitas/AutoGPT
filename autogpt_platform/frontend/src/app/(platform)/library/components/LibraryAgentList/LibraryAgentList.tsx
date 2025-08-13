"use client";
import LibraryActionSubHeader from "../LibraryActionSubHeader/LibraryActionSubHeader";
import LibraryAgentCard from "../LibraryAgentCard/LibraryAgentCard";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { useLibraryAgentList } from "./useLibraryAgentList";

export default function LibraryAgentList() {
  const {
    agentLoading,
    agentCount,
    allAgents: agents,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
  } = useLibraryAgentList();

  const LoadingSpinner = () => (
    <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-t-2 border-neutral-800" />
  );

  return (
    <>
      <LibraryActionSubHeader agentCount={agentCount} />
      <div className="px-2">
        {agentLoading ? (
          <div className="flex h-[200px] items-center justify-center">
            <LoadingSpinner />
          </div>
        ) : (
          <InfiniteScroll
            dataLength={agents.length}
            isFetchingNextPage={isFetchingNextPage}
            fetchNextPage={fetchNextPage}
            hasNextPage={hasNextPage}
            loader={<LoadingSpinner />}
          >
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
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
