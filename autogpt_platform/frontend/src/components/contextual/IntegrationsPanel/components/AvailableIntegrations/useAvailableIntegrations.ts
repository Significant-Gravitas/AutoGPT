"use client";

import { useEffect, useState } from "react";
import { useGetV1ListProviders } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import {
  filterConnectableProviders,
  toConnectableProviders,
} from "../ConnectServiceDialog/helpers";

const PAGE_SIZE = 24;

export function useAvailableIntegrations(query: string) {
  const [limit, setLimit] = useState(PAGE_SIZE);
  const debouncedQuery = useDebouncedValue(query, 250);
  const result = useGetV1ListProviders({
    query: {
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });
  const allProviders = toConnectableProviders(result.data ?? []);
  const matching = filterConnectableProviders(allProviders, debouncedQuery);
  // Paging applies whether or not there is a query. It used to be skipped
  // while searching, which is the one path a user reaches by typing, so a
  // broad term rendered every match and its image at once.
  const providers = matching.slice(0, limit);

  function showMore() {
    setLimit((current) => current + PAGE_SIZE);
  }

  useEffect(() => {
    setLimit(PAGE_SIZE);
  }, [debouncedQuery]);

  return {
    providers,
    total: matching.length,
    hasMore: providers.length < matching.length,
    showMore,
    isLoading: result.isLoading,
    isError: result.isError,
    error: result.error,
    refetch: result.refetch,
  };
}
