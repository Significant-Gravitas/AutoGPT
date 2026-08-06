"use client";

import { useGetBrainDumpRecommendedProviders } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import {
  useGetV1ListCredentials,
  useGetV1ListProviders,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import { useApiKeyConnectForm } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useApiKeyConnectForm";
import { useOAuthConnect } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useOAuthConnect";
import {
  AuthType,
  filterConnectableProviders,
  toConnectableProviders,
  type AuthMethod,
  type ConnectableProvider,
} from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/helpers";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useState } from "react";

const POLL_INTERVAL_MS = 2_500;
// ~1 minute of polling; a job the backend never finished (process restart
// mid-run) must not keep this dialog requesting forever.
const MAX_POLLS = 24;

export function useConnectToolsPanel() {
  // The endpoint 404s with the flag off, and a 404 never settles the poll
  // below on its own — so the request is not made at all.
  const isBrainDumpEnabled = useGetFlag(Flag.ONBOARDING_BRAIN_DUMP);
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query, 250);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [direction, setDirection] = useState<1 | -1>(1);
  const [selectedMethod, setSelectedMethod] = useState<AuthMethod | null>(null);

  const providersQuery = useGetV1ListProviders({
    query: {
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });

  // Connected state per provider; the OAuth and API-key flows invalidate
  // this query on success, so freshly connected rows flip immediately.
  const credentialsQuery = useGetV1ListCredentials({
    query: {
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });
  const connectedProviders = new Set(
    (credentialsQuery.data ?? []).map((credential) => credential.provider),
  );

  const recommendedQuery = useGetBrainDumpRecommendedProviders({
    query: {
      enabled: Boolean(isBrainDumpEnabled),
      refetchInterval: (q) => {
        const response = q.state.data;
        if (response && (response.status !== 200 || response.data.ready)) {
          return false;
        }
        // A failed request never reaches `data` — it rejects — so the
        // budget counts errors too; otherwise a 404 or a 500 would poll
        // for as long as the dialog stays open.
        const attempts = q.state.dataUpdateCount + q.state.errorUpdateCount;
        return attempts > MAX_POLLS ? false : POLL_INTERVAL_MS;
      },
    },
  });

  const allProviders = toConnectableProviders(providersQuery.data ?? []);
  const providers = filterConnectableProviders(allProviders, debouncedQuery);
  const selectedProvider: ConnectableProvider | null = selectedId
    ? (allProviders.find((provider) => provider.id === selectedId) ?? null)
    : null;

  const recommendations =
    recommendedQuery.data?.status === 200
      ? (recommendedQuery.data.data.providers ?? [])
      : [];
  // The model's reason replaces the generic provider description — that
  // line is what makes the section feel picked for this user. Unknown ids
  // (provider renamed or removed since the job ran) are dropped.
  const recommendedProviders = recommendations
    .map((recommendation): ConnectableProvider | null => {
      const provider = allProviders.find(
        (p) => p.id === recommendation.provider,
      );
      if (!provider) return null;
      return {
        ...provider,
        description: recommendation.reason || provider.description,
      };
    })
    .filter((provider) => provider !== null);

  function handleSelect(providerId: string) {
    setDirection(1);
    setSelectedId(providerId);
    setSelectedMethod(null);
    apiKey.form.reset();
  }

  function handleBackToList() {
    setDirection(-1);
    setSelectedId(null);
    setSelectedMethod(null);
  }

  // Completing a connection (OAuth or the API-key form) lands back on the
  // list so more tools can be wired up without leaving the dialog.
  const oauth = useOAuthConnect({
    provider: selectedProvider?.id ?? "",
    onSuccess: handleBackToList,
  });
  const apiKey = useApiKeyConnectForm({
    provider: selectedProvider?.id ?? "",
    onSuccess: handleBackToList,
  });

  function handleContinue() {
    if (selectedMethod === AuthType.oauth2) {
      oauth.connect();
      return;
    }
    if (selectedMethod === AuthType.api_key) {
      apiKey.form.handleSubmit(apiKey.handleSubmit)();
    }
  }

  return {
    query,
    setQuery,
    providers,
    recommendedProviders,
    isLoading: providersQuery.isLoading,
    isError: providersQuery.isError,
    error: providersQuery.error,
    refetch: providersQuery.refetch,
    selectedProvider,
    direction,
    connectedProviders,
    selectedMethod,
    setSelectedMethod,
    apiKeyForm: apiKey.form,
    handleApiKeySubmit: apiKey.handleSubmit,
    // The footer's Continue drives OAuth and the API-key form; the
    // remaining methods only show an unsupported notice.
    showContinue:
      !selectedMethod ||
      selectedMethod === AuthType.oauth2 ||
      selectedMethod === AuthType.api_key,
    isContinueDisabled:
      !selectedMethod ||
      (selectedMethod === AuthType.api_key && !apiKey.form.formState.isValid),
    isConnecting: oauth.isPending || apiKey.isPending,
    handleSelect,
    handleBackToList,
    handleContinue,
  };
}
