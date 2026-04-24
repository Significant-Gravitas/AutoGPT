"use client";

import { useState } from "react";

import {
  getGetV1ListUserApiKeysQueryKey,
  useGetV1ListUserApiKeys,
} from "@/app/api/__generated__/endpoints/api-keys/api-keys";
import { APIKeyStatus } from "@/app/api/__generated__/models/aPIKeyStatus";

export const API_KEYS_PAGE_SIZE = 15;

export const API_KEYS_PAGINATED_QUERY_KEY = getGetV1ListUserApiKeysQueryKey();

export function useAPIKeysList() {
  const [visiblePages, setVisiblePages] = useState(1);

  const query = useGetV1ListUserApiKeys({
    query: {
      select: (response) =>
        response.status === 200
          ? response.data.filter((key) => key.status === APIKeyStatus.ACTIVE)
          : [],
    },
  });

  const allKeys = query.data ?? [];
  const visibleCount = visiblePages * API_KEYS_PAGE_SIZE;
  const keys = allKeys.slice(0, visibleCount);
  const hasNextPage = allKeys.length > visibleCount;

  function fetchNextPage() {
    if (hasNextPage) setVisiblePages((n) => n + 1);
  }

  return {
    keys,
    totalCount: allKeys.length,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    hasNextPage,
    isFetchingNextPage: false,
    fetchNextPage,
    refetch: query.refetch,
    isEmpty: !query.isLoading && !query.isError && allKeys.length === 0,
  };
}
