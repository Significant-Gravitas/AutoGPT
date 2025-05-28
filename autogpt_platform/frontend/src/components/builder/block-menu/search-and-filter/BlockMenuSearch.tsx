import React, { useEffect, useState, useCallback, useRef } from "react";
import FiltersList from "./FiltersList";
import SearchList from "./SearchList";
import { useBlockMenuContext } from "../block-menu-provider";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

const BlockMenuSearch: React.FC = ({}) => {
  const {
    searchData,
    searchQuery,
    searchId,
    setSearchData,
    filters,
    setCategoryCounts,
  } = useBlockMenuContext();
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [hasMore, setHasMore] = useState<boolean>(true);
  const [page, setPage] = useState<number>(0);
  const [loadingMore, setLoadingMore] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const api = useBackendAPI();

  const pageSize = 10;

  const fetchSearchData = useCallback(
    async (pageNum: number, isLoadMore: boolean = false) => {
      if (isLoadMore) {
        setLoadingMore(true);
      } else {
        setIsLoading(true);
      }

      try {
        // Prepare filter array from active categories
        const activeCategories = Object.entries(filters.categories)
          .filter(([_, isActive]) => isActive)
          .map(([category, _]) => category)
          .filter((category) => category !== "templates") // API doesn't support templates filter
          .map(
            (category) =>
              category as
                | "blocks"
                | "integrations"
                | "marketplace_agents"
                | "my_agents",
          );

        const response = await api.searchBlocks({
          search_query: searchQuery,
          search_id: searchId,
          page: pageNum,
          page_size: pageSize,
          filter: activeCategories.length > 0 ? activeCategories : undefined,
          by_creator:
            filters.createdBy.length > 0 ? filters.createdBy : undefined,
        });

        setCategoryCounts(response.total_items);

        if (isLoadMore) {
          setSearchData((prev) => [...prev, ...response.items]);
        } else {
          setSearchData(response.items);
        }

        setHasMore(response.more_pages);
        setError(null);
      } catch (error) {
        console.error("Error fetching search data:", error);
        setError(
          error instanceof Error
            ? error.message
            : "Failed to load search results",
        );
        if (!isLoadMore) {
          setPage(0);
        }
      } finally {
        setIsLoading(false);
        setLoadingMore(false);
      }
    },
    [searchQuery, searchId, filters, api, setSearchData, pageSize],
  );

  const handleScroll = useCallback(() => {
    if (!scrollRef.current || loadingMore || !hasMore) return;

    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    if (scrollTop + clientHeight >= scrollHeight - 100) {
      const nextPage = page + 1;
      setPage(nextPage);
      fetchSearchData(nextPage, true);
    }
  }, [loadingMore, hasMore, page, fetchSearchData]);

  useEffect(() => {
    const scrollElement = scrollRef.current;
    if (scrollElement) {
      scrollElement.addEventListener("scroll", handleScroll);
      return () => scrollElement.removeEventListener("scroll", handleScroll);
    }
  }, [handleScroll]);

  useEffect(() => {
    if (searchQuery) {
      setPage(0);
      setHasMore(true);
      setError(null);
      fetchSearchData(0, false);
    } else {
      setSearchData([]);
      setError(null);
      setPage(0);
      setHasMore(true);
    }
  }, [searchQuery, searchId, filters, fetchSearchData, setSearchData]);

  return (
    <div
      ref={scrollRef}
      className="scrollbar-thumb-rounded h-full space-y-4 overflow-y-auto py-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200"
    >
      {searchData.length !== 0 && <FiltersList />}
      <SearchList
        isLoading={isLoading}
        loadingMore={loadingMore}
        hasMore={hasMore}
        error={error}
        onRetry={() => {
          setPage(0);
          setError(null);
          fetchSearchData(0, false);
        }}
      />
    </div>
  );
};

export default BlockMenuSearch;
