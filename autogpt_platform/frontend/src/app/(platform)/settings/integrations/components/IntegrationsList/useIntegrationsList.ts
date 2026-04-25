"use client";

import { useState } from "react";

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
import { useDebouncedValue } from "../hooks/useDebouncedValue";
import { useIntegrationsSelection } from "./useIntegrationsSelection";

export function useIntegrationsList() {
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query, 250);

  const credentialsQuery = useGetV1ListCredentials({
    query: {
      select: (response) =>
        response.status === 200 ? filterSystemCredentials(response.data) : [],
    },
  });

  const credentials = credentialsQuery.data ?? [];
  const allProviders: ProviderGroupView[] =
    groupCredentialsByProvider(credentials);
  const providers = filterProviders(allProviders, debouncedQuery);

  const allCredentialIds = providers.flatMap((p) =>
    p.credentials.map((c) => c.id),
  );
  const selection = useIntegrationsSelection(allCredentialIds);
  const {
    remove,
    isPending: isDeleting,
    isDeletingId,
  } = useDeleteIntegration();

  function buildTargets(ids: string[]): DeleteIntegrationTarget[] {
    const lookup = new Map<string, { provider: string; name: string }>();
    for (const provider of allProviders) {
      for (const cred of provider.credentials) {
        lookup.set(cred.id, { provider: cred.provider, name: cred.title });
      }
    }
    const targets: DeleteIntegrationTarget[] = [];
    for (const id of ids) {
      const entry = lookup.get(id);
      if (entry)
        targets.push({ id, provider: entry.provider, name: entry.name });
    }
    return targets;
  }

  async function requestDelete(ids: string[]) {
    const targets = buildTargets(ids);
    if (targets.length === 0) return;
    const result = await remove(targets);
    // Keep failed items selected so the user can retry without re-selecting.
    if (result.failed.length === 0) {
      selection.clear();
    } else {
      const failedIds = new Set(result.failed.map((t) => t.id));
      for (const id of ids) {
        if (!failedIds.has(id) && selection.isSelected(id)) {
          selection.toggle(id);
        }
      }
    }
  }

  const isLoading = credentialsQuery.isLoading;
  const isError = credentialsQuery.isError;
  const hasNoCredentials = !isLoading && !isError && allProviders.length === 0;

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
    isDeletingId,
    buildTargets,
  };
}
