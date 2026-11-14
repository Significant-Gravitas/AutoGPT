import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { useListExpertCredentials } from "@/app/api/__generated__/endpoints/experts/experts";
import { connectedIntegrationsFromCredentials } from "./helpers";

export function useConnectedIntegrations(expertId?: string | null) {
  const credentials = useGetV1ListCredentials({
    query: {
      enabled: !expertId,
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });
  const grants = useListExpertCredentials(expertId ?? "", {
    query: {
      enabled: Boolean(expertId),
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });
  if (expertId) {
    if (grants.isError) return [];
    return connectedIntegrationsFromCredentials(
      (grants.data ?? []).map((credential) => ({
        id: credential.credential_id,
        provider: credential.provider,
        title: credential.title,
        username: null,
      })),
    );
  }
  return connectedIntegrationsFromCredentials(
    credentials.isError ? [] : (credentials.data ?? []),
  );
}
