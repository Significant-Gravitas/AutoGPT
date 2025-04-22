"use client";
import { useEffect, useState, useCallback } from "react";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";

import { useLibraryPageContext } from "@/app/library/state-provider";
import { useScrollThreshold } from "@/hooks/useScrollThreshold";
import LibraryAgentCard from "./library-agent-card";

/**
 * Displays a grid of library agents with infinite scroll functionality.
 */
export default function LibraryAgentList(): React.ReactNode {
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);

  const api = useBackendAPI();
  const { agents, setAgents, setAgentLoading, agentLoading } =
    useLibraryPageContext();

  const fetchAgents = useCallback(
    async (page: number) => {
      try {
        const response = await api.listLibraryAgents(
          page === 1 ? {} : { page: page },
        );
        if (page > 1) {
          setAgents((prevAgent) => [...prevAgent, ...response.agents]);
        } else {
          setAgents(response.agents);
        }
        setHasMore(
          response.pagination.current_page * response.pagination.page_size <
            response.pagination.total_items,
        );
      } finally {
        setAgentLoading(false);
        setLoadingMore(false);
      }
    },
    [api, setAgents, setAgentLoading],
  );

  useEffect(() => {
    fetchAgents(1);
  }, [fetchAgents]);

  const handleInfiniteScroll = useCallback(
    (scrollY: number) => {
      if (!hasMore || loadingMore) return;

      const { scrollHeight, clientHeight } = document.documentElement;
      const SCROLL_THRESHOLD = 20;
      const FETCH_DELAY = 1000;

      if (scrollY + clientHeight >= scrollHeight - SCROLL_THRESHOLD) {
        setLoadingMore(true);
        const nextPage = currentPage + 1;
        setCurrentPage(nextPage);
        setTimeout(() => fetchAgents(nextPage), FETCH_DELAY);
      }
    },
    [currentPage, hasMore, loadingMore, fetchAgents],
  );

  useScrollThreshold(handleInfiniteScroll, 50);

  const LoadingSpinner = () => (
    <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-t-2 border-neutral-800" />
  );

  return (
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
          {loadingMore && hasMore && (
            <div className="flex items-center justify-center py-4 pt-8">
              <LoadingSpinner />
            </div>
          )}
        </>
      )}
    </div>
  );
}
