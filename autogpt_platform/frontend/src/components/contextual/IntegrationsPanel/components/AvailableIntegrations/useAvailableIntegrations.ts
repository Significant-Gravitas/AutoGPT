"use client";

import { useState } from "react";
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
  const providers = debouncedQuery.trim() ? matching : matching.slice(0, limit);

  function showMore() {
    setLimit((current) => current + PAGE_SIZE);
  }

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
