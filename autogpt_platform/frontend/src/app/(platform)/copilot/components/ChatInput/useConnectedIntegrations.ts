import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { connectedIntegrationsFromCredentials } from "./helpers";

/** The integrations a prompt can @-mention: one entry per provider the user
 *  has connected. Empty while disabled, loading, or when nothing is connected.
 *  Disabling only pauses fetching, so the shared credentials cache (filled by
 *  e.g. the Connections page) is ignored explicitly while disabled. */
export function useConnectedIntegrations(enabled: boolean) {
  const credentials = useGetV1ListCredentials({
    query: {
      enabled,
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });
  if (!enabled) return [];
  return connectedIntegrationsFromCredentials(credentials.data ?? []);
}
