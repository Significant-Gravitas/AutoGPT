import { useGetV1ListCredentialsByProvider } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { useOAuthConnect } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useOAuthConnect";
import {
  GOOGLE_ACCESS_SCOPES,
  GOOGLE_PROVIDER,
  type GoogleService,
  missingServices,
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

  // Services left unticked on Google's consent screen are asked for again,
  // added to the same account.
  function allow(services: readonly GoogleService[]) {
    if (!credential || services.length === 0) return;
    void oauth.connectWith({
      scopes: services.flatMap((service) => service.scopes),
      credentialID: credential.id,
    });
  }

  return {
    isLoading: saved.isLoading,
    // Not the same as "not connected": connecting again from there could add
    // a second connection for the same account.
    lookupFailed: saved.isError,
    lookupError: saved.error,
    retry: () => void saved.refetch(),
    credential,
    missing: credential ? missingServices(credential) : [],
    isPending: oauth.isPending,
    connect,
    allow,
  };
}
