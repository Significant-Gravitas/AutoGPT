import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useGetV1ListCredentialsByProvider } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { useOAuthConnect } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useOAuthConnect";
import {
  getMicrosoftSetupReason,
  MICROSOFT_PROVIDER,
  type MicrosoftSetupReason,
} from "@/lib/microsoft-setup";
import {
  getMicrosoftAccessQueryKey,
  useMicrosoftAccessCheck,
} from "./useMicrosoftAccessCheck";
import { pendingScopes } from "./helpers";

interface Args {
  /** What a first sign-in asks for. Omit for the basics. */
  scopes?: string[];
}

export function useMicrosoftConnection({ scopes }: Args) {
  const queryClient = useQueryClient();
  const saved = useGetV1ListCredentialsByProvider(MICROSOFT_PROVIDER);
  const credentials = saved.data?.status === 200 ? saved.data.data : [];
  const [selectedID, setSelectedID] = useState("");
  const [recovery, setRecovery] = useState<MicrosoftSetupReason | null>(null);
  const credential =
    credentials.find((item) => item.id === selectedID) ?? credentials[0];
  const access = useMicrosoftAccessCheck(credential?.id);
  const check = access.data ?? undefined;

  const oauth = useOAuthConnect({
    provider: MICROSOFT_PROVIDER,
    scopes,
    onSuccess: handleConnected,
    onError: handleError,
  });

  function handleError(error: unknown) {
    setRecovery(getMicrosoftSetupReason(error) ?? "sign_in_failed");
  }

  async function handleConnected(value?: CredentialsMetaResponse) {
    setRecovery(null);
    await saved.refetch();
    if (!value) return;
    setSelectedID(value.id);
    await queryClient.invalidateQueries({
      queryKey: getMicrosoftAccessQueryKey(value.id),
    });
  }

  // A new attempt starts clean: the last one's error no longer applies.
  function signIn() {
    setRecovery(null);
    void oauth.connect();
  }

  function allow(requested: string[]) {
    if (!credential || requested.length === 0) return;
    setRecovery(null);
    void oauth.connectWith({ scopes: requested, credentialID: credential.id });
  }

  function recheck() {
    void access.refetch();
  }

  return {
    credentials,
    credential,
    check,
    checking: access.isFetching,
    isLoading: saved.isLoading,
    pending: pendingScopes(check),
    recovery,
    oauth,
    signIn,
    allow,
    recheck,
    select: setSelectedID,
    handleConnected,
    handleError,
  };
}
