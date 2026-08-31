import {
  getListExpertCredentialsQueryKey,
  useGrantExpertCredentials,
  useListExpertCredentials,
  useRevokeExpertCredential,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
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
  // Credentials already on the account when the connect dialog opened. Null
  // means we could not read them, so a freshly created credential cannot be
  // told apart from a pre-existing one and auto-granting would hand the expert
  // access the user never picked.
  const connectedBefore = useRef<Set<string> | null>(null);

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
    connectedBefore.current = connectedQuery.isSuccess
      ? new Set((connectedQuery.data ?? []).map((c) => c.id))
      : null;
    setIsConnecting(true);
  }

  async function closeConnect() {
    setIsConnecting(false);
    const before = connectedBefore.current;
    connectedBefore.current = null;

    // Every connect method — OAuth, API key, device code, MCP — lands in the
    // same credentials list, so diffing it picks up whatever was just created
    // without threading a new id back out of each individual flow.
    const { data } = await connectedQuery.refetch();
    const created = before
      ? (data ?? []).filter((c) => !before.has(c.id)).map((c) => c.id)
      : [];

    // Nothing was connected (or we cannot tell): drop the user back where they
    // were instead of silently swallowing the click.
    if (created.length === 0) {
      setIsAdding(true);
      return;
    }
    grant({ expertId, data: { credential_ids: created } });
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
    addIntegration,
    removeIntegration,
    isGranting,
    isRevoking,
  };
}
