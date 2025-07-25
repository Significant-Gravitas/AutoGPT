"use client";
import LibraryActionSubHeader from "../LibraryActionSubHeader/LibraryActionSubHeader";
import LibraryAgentCard from "../LibraryAgentCard/LibraryAgentCard";
import { useLibraryAgentList } from "./useLibraryAgentList";

export default function LibraryAgentList() {
  const {
    agentLoading,
    agentCount,
    allAgents: agents,
    isFetchingNextPage,
    isSearching,
  } = useLibraryAgentList();

  const LoadingSpinner = () => (
    <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-t-2 border-neutral-800" />
  );

  return (
    <>
      {/* TODO: We need a new endpoint on backend that returns total number of agents */}
      <LibraryActionSubHeader agentCount={agentCount} />
      <div className="px-2">
        {agentLoading ? (
          <div className="flex h-[200px] items-center justify-center">
            <LoadingSpinner />
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {agents.map((agent) => (
                <LibraryAgentCard key={agent.id} agent={agent} />
              ))}
            </div>
            {(isFetchingNextPage || isSearching) && (
              <div className="flex items-center justify-center py-4 pt-8">
                <LoadingSpinner />
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}
