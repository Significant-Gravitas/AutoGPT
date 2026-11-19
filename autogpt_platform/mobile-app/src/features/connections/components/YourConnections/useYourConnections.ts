import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { integrationKeys, listCredentials } from "@/lib/api/integrations";

import { accountRows } from "../../listHelpers";

export function useYourConnections(isOpen: boolean) {
  const [isAdding, setIsAdding] = useState(false);
  const credentials = useQuery({
    queryKey: integrationKeys.credentials,
    queryFn: listCredentials,
    enabled: isOpen,
  });

  function openAdd() {
    setIsAdding(true);
  }

  function closeAdd() {
    setIsAdding(false);
  }

  function retry() {
    void credentials.refetch();
  }

  return {
    accounts: accountRows(credentials.data ?? []),
    isLoading: !credentials.data && credentials.isPending,
    hasFailed: !credentials.data && credentials.isError,
    retry,
    isAdding,
    openAdd,
    closeAdd,
  };
}
