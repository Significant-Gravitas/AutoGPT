import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { useState } from "react";
import { filterCredentialsByProvider } from "./helpers";

export const useCredentialField = ({
  credentialSchema,
}: {
  credentialSchema: BlockIOCredentialsSubSchema; // Here we are using manual typing, we need to fix it with automatic one
}) => {
  const [isAPIKeyModalOpen, setIsAPIKeyModalOpen] = useState(false);

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

  const supportsApiKey = credentialSchema.credentials_types.includes("api_key");
  const supportsOAuth2 = credentialSchema.credentials_types.includes("oauth2");

  const credentialProviders = credentialSchema.credentials_provider;
  const { credentials: filteredCredentials, exists: credentialsExists } =
    filterCredentialsByProvider(credentials, credentialProviders);

  return {
    credentials: filteredCredentials,
    isCredentialListLoading,
    supportsApiKey,
    supportsOAuth2,
    isAPIKeyModalOpen,
    setIsAPIKeyModalOpen,
    credentialsExists,
  };
};
