import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";

export const useCredentialField = () => {
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
  return {
    credentials,
    isCredentialListLoading,
  };
};
