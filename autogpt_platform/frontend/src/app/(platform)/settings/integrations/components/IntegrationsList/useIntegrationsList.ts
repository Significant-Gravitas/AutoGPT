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

  const allCredentialIds = allProviders.flatMap((p) =>
    p.credentials.filter((c) => !c.isManaged).map((c) => c.id),
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

  async function requestDelete(ids: string[], force = false) {
    const targets = buildTargets(ids);
    if (targets.length === 0) return { needsConfirmationIds: [] as string[] };
    const result = await remove(targets, force);
    const needsConfirmationIds = result.needsConfirmation.map(
      (c) => c.target.id,
    );
    // Keep failed AND needs-confirmation items selected so the user retains
    // visual context of which credentials still need action.
    const keepSelected = new Set([
      ...result.failed.map((t) => t.id),
      ...needsConfirmationIds,
    ]);
    if (keepSelected.size === 0) {
      selection.clear();
      return { needsConfirmationIds };
    }
    for (const id of ids) {
      if (!keepSelected.has(id) && selection.isSelected(id)) {
        selection.toggle(id);
      }
    }
    return { needsConfirmationIds };
  }

  const isLoading = credentialsQuery.isLoading;
  const isError = credentialsQuery.isError;

  return {
    query,
    setQuery,
    providers,
    isLoading,
    isError,
    error: credentialsQuery.error,
    refetch: credentialsQuery.refetch,
    isEmpty: providers.length === 0,
    selection,
    requestDelete,
    isDeleting,
    isDeletingId,
    buildTargets,
  };
}
