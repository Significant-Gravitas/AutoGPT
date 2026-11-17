import { useGetV1ListCredentialsByProvider } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { useOAuthConnect } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useOAuthConnect";
import {
  GOOGLE_ACCESS_SCOPES,
  GOOGLE_PROVIDER,
  type GoogleService,
  pickGoogleCredential,
} from "./helpers";

export function useGoogleAccess() {
  const saved = useGetV1ListCredentialsByProvider(GOOGLE_PROVIDER);
  const credentials = saved.data?.status === 200 ? saved.data.data : [];
  const credential = pickGoogleCredential(credentials);

  const oauth = useOAuthConnect({
    provider: GOOGLE_PROVIDER,
    scopes: GOOGLE_ACCESS_SCOPES,
    onSuccess: handleConnected,
  });

  async function handleConnected() {
    await saved.refetch();
  }

  function connect() {
    void oauth.connect();
  }

  // A service left unticked on Google's consent screen is asked for again,
  // added to the same account.
  function allow(service: GoogleService) {
    if (!credential) return;
    void oauth.connectWith({
      scopes: [...service.scopes],
      credentialID: credential.id,
    });
  }

  return {
    isLoading: saved.isLoading,
    credential,
    pending: oauth.isPending,
    connect,
    allow,
  };
}
