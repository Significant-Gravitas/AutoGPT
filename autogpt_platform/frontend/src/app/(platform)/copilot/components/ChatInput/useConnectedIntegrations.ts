import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { connectedIntegrationsFromCredentials } from "./helpers";

/** The integrations a prompt can @-mention: one entry per provider the user
 *  has connected. Empty while disabled, loading, or when nothing is connected. */
export function useConnectedIntegrations(enabled: boolean) {
  const credentials = useGetV1ListCredentials({
    query: {
      enabled,
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });
  return connectedIntegrationsFromCredentials(credentials.data ?? []);
}
