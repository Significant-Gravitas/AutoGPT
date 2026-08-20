"use client";

import { useApiKeyConnectForm } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useApiKeyConnectForm";
import { useOAuthConnect } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useOAuthConnect";
import {
  AuthType,
  type AuthMethod,
} from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/helpers";
import { useState } from "react";

interface Args {
  provider: string;
  onConnected: () => void;
}

export function useConnectCredentialDialog({ provider, onConnected }: Args) {
  const [selectedMethod, setSelectedMethod] = useState<AuthMethod | null>(null);

  const oauth = useOAuthConnect({ provider, onSuccess: handleConnected });
  const apiKey = useApiKeyConnectForm({ provider, onSuccess: handleConnected });

  // The dialog stays mounted while closed, so a half-filled key or a
  // picked method would still be there the next time it opens.
  function reset() {
    setSelectedMethod(null);
    apiKey.form.reset();
  }

  function handleConnected() {
    reset();
    onConnected();
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
    selectedMethod,
    setSelectedMethod,
    apiKeyForm: apiKey.form,
    handleApiKeySubmit: apiKey.handleSubmit,
    // Continue drives OAuth and the API-key form; the remaining methods
    // only show an unsupported notice.
    showContinue:
      !selectedMethod ||
      selectedMethod === AuthType.oauth2 ||
      selectedMethod === AuthType.api_key,
    isContinueDisabled:
      !selectedMethod ||
      (selectedMethod === AuthType.api_key && !apiKey.form.formState.isValid),
    isConnecting: oauth.isPending || apiKey.isPending,
    handleContinue,
    reset,
  };
}
