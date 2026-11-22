"use client";

import { useEffect, useState } from "react";

import { useGetV1ListProviders } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";

import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import {
  filterConnectableProviders,
  popularProviders,
  sortByPopularity,
  toConnectableProviders,
  withoutProviders,
  type ConnectableProvider,
} from "./helpers";

interface Args {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConnected?: (credential: CredentialsMetaResponse) => void;
  initialProviderId?: string | null;
  /** Opens with the service search already filled in. */
  initialQuery?: string;
}

export function useConnectServiceDialog({
  open,
  onOpenChange,
  onConnected,
  initialProviderId,
  initialQuery,
}: Args) {
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query, 250);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [direction, setDirection] = useState<1 | -1>(1);

  const providersQuery = useGetV1ListProviders({
    query: {
      enabled: open,
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });

  useEffect(() => {
    if (open) {
      setSelectedId(initialProviderId ?? null);
      setQuery(initialQuery ?? "");
    } else {
      setQuery("");
      setSelectedId(null);
      setDirection(1);
    }
  }, [open, initialProviderId, initialQuery]);

  // The search it opens with filters at once; only typing waits on the
  // debounce, so the full list never flashes before the filtered one.
  const filterQuery =
    initialQuery && query === initialQuery ? query : debouncedQuery;
  const allProviders = toConnectableProviders(providersQuery.data ?? []);
  const isSearching = filterQuery.trim().length > 0;
  const popular = isSearching ? [] : popularProviders(allProviders);
  const providers = isSearching
    ? sortByPopularity(filterConnectableProviders(allProviders, filterQuery))
    : sortByPopularity(withoutProviders(allProviders, popular));

  const selectedProvider: ConnectableProvider | null = selectedId
    ? (allProviders.find((p) => p.id === selectedId) ?? null)
    : null;

  const view: "list" | "detail" = selectedProvider ? "detail" : "list";

  function handleSelect(providerId: string) {
    setDirection(1);
    setSelectedId(providerId);
  }

  function handleBack() {
    setDirection(-1);
    setSelectedId(null);
  }

  // Report the credential this dialog created before closing, so callers act
  // on the one they asked for rather than on whatever else appeared meanwhile.
  function handleSuccess(credential?: CredentialsMetaResponse) {
    if (credential) onConnected?.(credential);
    onOpenChange(false);
  }

  return {
    query,
    setQuery,
    providers,
    popular,
    isLoading: providersQuery.isLoading,
    isError: providersQuery.isError,
    error: providersQuery.error,
    refetch: providersQuery.refetch,
    view,
    direction,
    selectedProvider,
    handleSelect,
    handleBack,
    handleSuccess,
  };
}
