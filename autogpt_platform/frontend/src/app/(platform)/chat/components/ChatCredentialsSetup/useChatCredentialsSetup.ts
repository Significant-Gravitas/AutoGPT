import { useState, useEffect, useMemo, useContext } from "react";
import { toast } from "sonner";
import type { CredentialInfo } from "./ChatCredentialsSetup";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";

interface CredentialStatus {
  isConfigured: boolean;
  credentialId?: string;
}

interface ActiveModal {
  index: number;
  type: "api_key" | "oauth2" | "user_password" | "host_scoped";
  provider: string;
  providerName: string;
  schema?: BlockIOCredentialsSubSchema;
}

export function useChatCredentialsSetup(credentials: CredentialInfo[]) {
  const allProviders = useContext(CredentialsProvidersContext);
  const [credentialStatuses, setCredentialStatuses] = useState<
    CredentialStatus[]
  >([]);
  const [activeModal, setActiveModal] = useState<ActiveModal | null>(null);

  // Check existing credentials on mount
  useEffect(
    function checkExistingCredentials() {
      const statuses = credentials.map((cred) => {
        const provider = allProviders?.[cred.provider];
        const hasSavedCredentials =
          provider && provider.savedCredentials && provider.savedCredentials.length > 0;

        return {
          isConfigured: hasSavedCredentials || false,
          credentialId: hasSavedCredentials ? provider.savedCredentials[0].id : undefined,
        };
      });
      setCredentialStatuses(statuses);
    },
    [credentials, allProviders]
  );

  const isAllComplete = useMemo(
    function checkAllComplete() {
      if (credentialStatuses.length === 0) return false;
      return credentialStatuses.every((status) => status.isConfigured);
    },
    [credentialStatuses]
  );

  function handleSetupClick(index: number, credential: CredentialInfo) {
    const provider = allProviders?.[credential.provider];

    if (!provider) {
      toast.error("Provider not found", {
        description: `Unable to find configuration for ${credential.providerName}`,
      });
      return;
    }

    // Create a minimal schema for the modal
    const schema: BlockIOCredentialsSubSchema = {
      type: "object",
      properties: {},
      credentials_provider: [credential.provider],
      credentials_types: [credential.credentialType],
      credentials_scopes: credential.scopes,
      discriminator: undefined,
      discriminator_mapping: undefined,
      discriminator_values: undefined,
    };

    setActiveModal({
      index,
      type: credential.credentialType,
      provider: credential.provider,
      providerName: credential.providerName,
      schema,
    });
  }

  function handleModalClose() {
    setActiveModal(null);
  }

  function handleCredentialCreated(credentialMeta: CredentialsMetaInput) {
    if (activeModal) {
      // Mark credential as complete
      setCredentialStatuses((prev) => {
        const updated = [...prev];
        updated[activeModal.index] = {
          isConfigured: true,
          credentialId: credentialMeta.id,
        };
        return updated;
      });

      toast.success("Credential added successfully", {
        description: `${activeModal.providerName} credentials have been configured`,
      });
      setActiveModal(null);
    }
  }

  return {
    credentialStatuses,
    isAllComplete,
    activeModal,
    handleSetupClick,
    handleModalClose,
    handleCredentialCreated,
  };
}
