"use client";

import { useEffect, useMemo, useState } from "react";

import { useGetV1ListProviders } from "@/app/api/__generated__/endpoints/integrations/integrations";

import { formatProviderName } from "../../helpers";
import {
  filterConnectableProviders,
  toConnectableProviders,
  type ConnectableProvider,
} from "./helpers";

interface Args {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function useConnectServiceDialog({ open, onOpenChange }: Args) {
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [direction, setDirection] = useState<1 | -1>(1);

  const providersQuery = useGetV1ListProviders({
    query: {
      enabled: open,
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });

  useEffect(() => {
    if (!open) {
      setQuery("");
      setSelectedId(null);
      setDirection(1);
    }
  }, [open]);

  const allProviders = useMemo(
    () => toConnectableProviders(providersQuery.data ?? []),
    [providersQuery.data],
  );

  const providers = useMemo(
    () => filterConnectableProviders(allProviders, query),
    [allProviders, query],
  );

  const selectedProvider: ConnectableProvider | null = selectedId
    ? allProviders.find((p) => p.id === selectedId) ??
      (providersQuery.data?.some((p) => p.name === selectedId)
        ? { id: selectedId, name: formatProviderName(selectedId) }
        : null)
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

  function handleSuccess() {
    onOpenChange(false);
  }

  return {
    query,
    setQuery,
    providers,
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
