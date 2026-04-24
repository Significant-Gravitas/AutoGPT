"use client";

import { useMemo, useState } from "react";

import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { filterSystemCredentials } from "@/components/contextual/CredentialsInput/helpers";

import {
  filterProviders,
  groupCredentialsByProvider,
  type ProviderGroupView,
} from "../../helpers";
import {
  useDeleteIntegration,
  type DeleteIntegrationTarget,
} from "../hooks/useDeleteIntegration";
import { useIntegrationsSelection } from "./useIntegrationsSelection";

export function useIntegrationsList() {
  const [query, setQuery] = useState("");

  const credentialsQuery = useGetV1ListCredentials({
    query: {
      select: (response) =>
        response.status === 200
          ? filterSystemCredentials(response.data)
          : [],
    },
  });

  const credentials = credentialsQuery.data ?? [];

  const allProviders: ProviderGroupView[] = useMemo(
    () => groupCredentialsByProvider(credentials),
    [credentials],
  );

  const providers = useMemo(
    () => filterProviders(allProviders, query),
    [allProviders, query],
  );

  const allCredentialIds = providers.flatMap((p) =>
    p.credentials.map((c) => c.id),
  );
  const selection = useIntegrationsSelection(allCredentialIds);
  const { remove, isPending: isDeleting } = useDeleteIntegration();

  function buildTargets(ids: string[]): DeleteIntegrationTarget[] {
    const lookup = new Map<string, string>();
    for (const provider of allProviders) {
      for (const cred of provider.credentials) {
        lookup.set(cred.id, cred.provider);
      }
    }
    const targets: DeleteIntegrationTarget[] = [];
    for (const id of ids) {
      const provider = lookup.get(id);
      if (provider) targets.push({ id, provider });
    }
    return targets;
  }

  async function requestDelete(ids: string[]) {
    const targets = buildTargets(ids);
    if (targets.length === 0) return;
    await remove(targets);
    selection.clear();
  }

  const isLoading = credentialsQuery.isLoading;
  const isError = credentialsQuery.isError;
  const hasNoCredentials =
    !isLoading && !isError && allProviders.length === 0;

  return {
    query,
    setQuery,
    providers,
    isLoading,
    isError,
    error: credentialsQuery.error,
    refetch: credentialsQuery.refetch,
    isEmpty: providers.length === 0,
    hasNoCredentials,
    selection,
    requestDelete,
    isDeleting,
  };
}
