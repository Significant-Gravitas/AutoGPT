import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import {
  filterCredentialsByProvider,
  getCredentialProviderFromSchema,
} from "./helpers";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useEffect, useRef } from "react";
import { useShallow } from "zustand/react/shallow";

export const useCredentialField = ({
  credentialSchema,
  formData,
  nodeId,
  onChange,
}: {
  credentialSchema: BlockIOCredentialsSubSchema; // Here we are using manual typing, we need to fix it with automatic one
  formData: Record<string, any>;
  nodeId: string;
  onChange: (value: Record<string, any>) => void;
}) => {
  const previousProviderRef = useRef<string | null>(null);

  // Fetch all the credentials from the backend
  // We will save it in cache for 10 min, if user edits the credential, we will invalidate the cache
  // Whenever user adds a block, we filter the credentials list and check if this block's provider is in the list
  const { data: credentials, isLoading: isCredentialListLoading } =
    useGetV1ListCredentials({
      query: {
        refetchInterval: 10 * 60 * 1000,
        select: (x) => {
          return x.data as CredentialsMetaResponse[];
        },
      },
    });

  const hardcodedValues = useNodeStore(
    useShallow((state) => state.getHardCodedValues(nodeId)),
  );

  const credentialProvider = getCredentialProviderFromSchema(
    hardcodedValues,
    credentialSchema,
  );

  const supportsApiKey = credentialSchema.credentials_types.includes("api_key");
  const supportsOAuth2 = credentialSchema.credentials_types.includes("oauth2");
  const supportsUserPassword =
    credentialSchema.credentials_types.includes("user_password");

  const { credentials: filteredCredentials, exists: credentialsExists } =
    filterCredentialsByProvider(credentials, credentialProvider ?? "");

  const setCredential = (credentialId: string) => {
    const selectedCredential = filteredCredentials.find(
      (c) => c.id === credentialId,
    );
    if (selectedCredential) {
      onChange({
        ...formData,
        id: selectedCredential.id,
        provider: selectedCredential.provider,
        title: selectedCredential.title,
        type: selectedCredential.type,
      });
    }
  };

  // This side effect is used to clear the hardcoded value in credential formData when the provider changes
  useEffect(() => {
    if (!credentialProvider) return;
    // If provider has changed and we have a credential selected
    if (
      previousProviderRef.current !== null &&
      previousProviderRef.current !== credentialProvider &&
      formData.id
    ) {
      // Check if the current credential belongs to the new provider
      const currentCredentialBelongsToProvider = filteredCredentials.some(
        (c) => c.id === formData.id,
      );

      // If not, clear the credential
      if (!currentCredentialBelongsToProvider) {
        onChange({
          id: "",
          provider: "",
          title: "",
          type: "",
        });
      }
    }
    previousProviderRef.current = credentialProvider;
  }, [credentialProvider, formData.id, credentials, onChange]);

  // This side effect is used to auto-select the latest credential when none is selected [latest means last one in the list of credentials]
  useEffect(() => {
    if (
      !isCredentialListLoading &&
      filteredCredentials.length > 0 &&
      !formData.id && // No credential currently selected
      credentialProvider // Provider is set
    ) {
      const latestCredential =
        filteredCredentials[filteredCredentials.length - 1];
      setCredential(latestCredential.id);
    }
  }, [
    isCredentialListLoading,
    filteredCredentials.length,
    formData.id,
    credentialProvider,
  ]);

  return {
    credentials: filteredCredentials,
    isCredentialListLoading,
    setCredential,
    supportsApiKey,
    supportsOAuth2,
    supportsUserPassword,
    credentialsExists,
    credentialProvider,
  };
};
