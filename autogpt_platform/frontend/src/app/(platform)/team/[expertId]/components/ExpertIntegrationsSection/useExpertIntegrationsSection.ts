import {
  getListExpertCredentialsQueryKey,
  useGrantExpertCredentials,
  useListExpertCredentials,
  useRevokeExpertCredential,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { okData } from "@/app/api/helpers";
import { filterSystemCredentials } from "@/components/contextual/CredentialsInput/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";

export function useExpertIntegrationsSection(expertId: string) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [isAdding, setIsAdding] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  // Whether the dialog we just closed actually created something. Closing
  // without a credential means the user backed out, and the picker they came
  // from should come back rather than the click going nowhere.
  const didConnect = useRef(false);

  const grantedQuery = useListExpertCredentials(expertId, {
    query: { select: (response) => okData(response) ?? [] },
  });
  // System credentials (platform LLM keys) are filtered out because they are
  // never granted and never revocable — offering them would suggest otherwise.
  const connectedQuery = useGetV1ListCredentials({
    query: {
      select: (response) =>
        response.status === 200 ? filterSystemCredentials(response.data) : [],
    },
  });

  const granted = grantedQuery.data ?? [];
  const grantedIds = new Set(granted.map((g) => g.credential_id));
  const grantable = (connectedQuery.data ?? []).filter(
    (credential) => !grantedIds.has(credential.id),
  );

  function invalidate() {
    queryClient.invalidateQueries({
      queryKey: getListExpertCredentialsQueryKey(expertId),
    });
  }

  const { mutate: grant, isPending: isGranting } = useGrantExpertCredentials({
    mutation: {
      onSuccess: () => {
        setIsAdding(false);
        invalidate();
      },
      onError: () =>
        toast({ title: "Could not add integration", variant: "destructive" }),
    },
  });

  const { mutate: revoke, isPending: isRevoking } = useRevokeExpertCredential({
    mutation: {
      onSuccess: invalidate,
      onError: () =>
        toast({
          title: "Could not remove integration",
          variant: "destructive",
        }),
    },
  });

  function openAdd() {
    setIsAdding(true);
  }

  function closeAdd() {
    setIsAdding(false);
  }

  function openConnect() {
    setIsAdding(false);
    didConnect.current = false;
    setIsConnecting(true);
  }

  // The dialog names the credential it created, so a credential the user
  // happens to add elsewhere while it is open is never swept in.
  function connectCredential(credential: CredentialsMetaResponse) {
    didConnect.current = true;
    grant({ expertId, data: { credential_ids: [credential.id] } });
  }

  function closeConnect() {
    setIsConnecting(false);
    if (!didConnect.current) setIsAdding(true);
    didConnect.current = false;
  }

  function addIntegration(credentialId: string) {
    grant({ expertId, data: { credential_ids: [credentialId] } });
  }

  function removeIntegration(credentialId: string) {
    revoke({ expertId, credentialId });
  }

  async function refetch() {
    await Promise.all([grantedQuery.refetch(), connectedQuery.refetch()]);
  }

  return {
    granted,
    grantable,
    // Both queries are reported separately: a failed connected-credentials read
    // must not claim the expert has no access, and a failed granted read must
    // not claim there is nothing left to add. Collapsing either into an empty
    // array would render a confident lie.
    isLoading: grantedQuery.isLoading,
    isError: grantedQuery.isError,
    isGrantableLoading: connectedQuery.isLoading,
    isGrantableError: connectedQuery.isError,
    refetch,
    isAdding,
    openAdd,
    closeAdd,
    isConnecting,
    openConnect,
    closeConnect,
    connectCredential,
    addIntegration,
    removeIntegration,
    isGranting,
    isRevoking,
  };
}
