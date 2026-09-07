import {
  useGetV1ListCredentials,
  useGetV1ListProviders,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
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
import { useEffect, useState } from "react";

interface Args {
  open: boolean;
  onConnected: (credential: CredentialsMetaResponse) => void;
}

export function useExpertConnectServiceDialog({ open, onConnected }: Args) {
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query, 250);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedMethod, setSelectedMethod] = useState<AuthMethod | null>(null);
  const [direction, setDirection] = useState<1 | -1>(1);

  const providersQuery = useGetV1ListProviders({
    query: {
      enabled: open,
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });
  const credentialsQuery = useGetV1ListCredentials({
    query: {
      enabled: open,
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });

  useEffect(() => {
    if (!open) {
      setQuery("");
      setSelectedId(null);
      setSelectedMethod(null);
    }
  }, [open]);

  const allProviders = toConnectableProviders(providersQuery.data ?? []);
  const credentials = credentialsQuery.data ?? [];
  const connectedProviders = new Set(
    credentials.map((credential) => credential.provider),
  );
  const providers = filterConnectableProviders(allProviders, debouncedQuery);
  const selectedProvider: ConnectableProvider | null = selectedId
    ? (allProviders.find((provider) => provider.id === selectedId) ?? null)
    : null;
  function handleSuccess(credential?: CredentialsMetaResponse) {
    setDirection(-1);
    setSelectedId(null);
    setSelectedMethod(null);
    if (credential) onConnected(credential);
  }

  const oauth = useOAuthConnect({
    provider: selectedProvider?.id ?? "",
    onSuccess: handleSuccess,
  });
  const apiKey = useApiKeyConnectForm({
    provider: selectedProvider?.id ?? "",
    onSuccess: handleSuccess,
  });

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
    isLoading: providersQuery.isLoading,
    isError: providersQuery.isError,
    refetch: providersQuery.refetch,
    selectedProvider,
    direction,
    connectedProviders,
    selectedMethod,
    setSelectedMethod,
    apiKeyForm: apiKey.form,
    handleApiKeySubmit: apiKey.handleSubmit,
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
    handleSuccess,
  };
}
