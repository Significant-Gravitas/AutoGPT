import { useState, useCallback, useRef, useEffect } from "react";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  Block,
  BlockRequest,
  Provider,
  StoreAgent,
  LibraryAgent,
  LibraryAgentSortEnum,
} from "@/lib/autogpt-server-api";

type BlocksPaginationRequest = { apiType: "blocks" } & BlockRequest;
type ProvidersPaginationRequest = { apiType: "providers" } & {
  page?: number;
  page_size?: number;
};
type StoreAgentsPaginationRequest = { apiType: "store-agents" } & {
  featured?: boolean;
  creator?: string;
  sorted_by?: string;
  search_query?: string;
  category?: string;
  page?: number;
  page_size?: number;
};
type LibraryAgentsPaginationRequest = { apiType: "library-agents" } & {
  search_term?: string;
  sort_by?: LibraryAgentSortEnum;
  page?: number;
  page_size?: number;
};

type PaginationRequest =
  | BlocksPaginationRequest
  | ProvidersPaginationRequest
  | StoreAgentsPaginationRequest
  | LibraryAgentsPaginationRequest;

interface UsePaginationOptions<T extends PaginationRequest> {
  request: T;
  pageSize?: number;
  enabled?: boolean;
}

interface UsePaginationReturn<T> {
  data: T[];
  loading: boolean;
  loadingMore: boolean;
  hasMore: boolean;
  error: string | null;
  scrollRef: React.RefObject<HTMLDivElement>;
  refresh: () => void;
  loadMore: () => void;
}

type GetReturnType<T> = T extends BlocksPaginationRequest
  ? Block
  : T extends ProvidersPaginationRequest
    ? Provider
    : T extends StoreAgentsPaginationRequest
      ? StoreAgent
      : T extends LibraryAgentsPaginationRequest
        ? LibraryAgent
        : never;

export const usePagination = <T extends PaginationRequest>({
  request,
  pageSize = 10,
  enabled = true, // to allow pagination or not
}: UsePaginationOptions<T>): UsePaginationReturn<GetReturnType<T>> => {
  const [data, setData] = useState<GetReturnType<T>[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const isLoadingRef = useRef(false);
  const requestRef = useRef(request);
  const api = useBackendAPI();

  // because we are using this pagination for multiple components
  requestRef.current = request;

  const fetchData = useCallback(
    async (page: number, isLoadMore = false) => {
      if (isLoadingRef.current || !enabled) return;

      isLoadingRef.current = true;

      if (isLoadMore) {
        setLoadingMore(true);
      } else {
        setLoading(true);
      }

      setError(null);

      try {
        let response;
        let newData: GetReturnType<T>[];
        let pagination;

        const currentRequest = requestRef.current;
        const requestWithPagination = {
          ...currentRequest,
          page,
          page_size: pageSize,
        };

        switch (currentRequest.apiType) {
          case "blocks":
            const { apiType: _, ...blockRequest } = requestWithPagination;
            response = await api.getBuilderBlocks(blockRequest);
            newData = response.blocks as GetReturnType<T>[];
            pagination = response.pagination;
            break;

          case "providers":
            const { apiType: __, ...providerRequest } = requestWithPagination;
            response = await api.getProviders(providerRequest);
            newData = response.providers as GetReturnType<T>[];
            pagination = response.pagination;
            break;

          case "store-agents":
            const { apiType: ___, ...storeAgentRequest } =
              requestWithPagination;
            response = await api.getStoreAgents(storeAgentRequest);
            newData = response.agents as GetReturnType<T>[];
            pagination = response.pagination;
            break;

          case "library-agents":
            const { apiType: ____, ...libraryAgentRequest } =
              requestWithPagination;
            response = await api.listLibraryAgents(libraryAgentRequest);
            newData = response.agents as GetReturnType<T>[];
            pagination = response.pagination;
            break;

          default:
            throw new Error(
              `Unknown request type: ${(currentRequest as any).apiType}`,
            );
        }

        if (isLoadMore) {
          setData((prev) => [...prev, ...newData]);
        } else {
          setData(newData);
        }

        setHasMore(page < pagination.total_pages);
        setCurrentPage(page);
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to fetch data";
        setError(errorMessage);
        console.error("Error fetching data:", err);
      } finally {
        setLoading(false);
        setLoadingMore(false);
        isLoadingRef.current = false;
      }
    },
    [api, pageSize, enabled],
  );

  const handleScroll = useCallback(() => {
    const scrollElement = scrollRef.current;
    if (
      !scrollElement ||
      loadingMore ||
      !hasMore ||
      isLoadingRef.current ||
      !enabled
    )
      return;

    const { scrollTop, scrollHeight, clientHeight } = scrollElement;
    const threshold = 100;

    if (scrollTop + clientHeight >= scrollHeight - threshold) {
      fetchData(currentPage + 1, true);
    }
  }, [fetchData, currentPage, loadingMore, hasMore, enabled]);

  const refresh = useCallback(() => {
    setCurrentPage(1);
    setHasMore(true);
    setError(null);
    fetchData(1);
  }, [fetchData]);

  const loadMore = useCallback(() => {
    if (!loadingMore && hasMore && !isLoadingRef.current && enabled) {
      fetchData(currentPage + 1, true);
    }
  }, [fetchData, currentPage, loadingMore, hasMore, enabled]);

  const requestString = JSON.stringify(request);

  useEffect(() => {
    if (enabled) {
      setCurrentPage(1);
      setHasMore(true);
      setError(null);
      setData([]);
      fetchData(1);
    }
  }, [requestString, enabled, fetchData]);

  useEffect(() => {
    const scrollElement = scrollRef.current;
    if (scrollElement && enabled) {
      scrollElement.addEventListener("scroll", handleScroll);
      return () => scrollElement.removeEventListener("scroll", handleScroll);
    }
  }, [handleScroll, enabled]);

  return {
    data,
    loading,
    loadingMore,
    hasMore,
    error,
    scrollRef,
    refresh,
    loadMore,
  };
};
