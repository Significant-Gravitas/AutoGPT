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
import { useState } from "react";

export function useExpertIntegrationsSection(expertId: string) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [isAdding, setIsAdding] = useState(false);

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

  return {
    granted,
    grantable,
    isLoading: grantedQuery.isLoading,
    isError: grantedQuery.isError,
    refetch: grantedQuery.refetch,
    isAdding,
    openAdd: () => setIsAdding(true),
    closeAdd: () => setIsAdding(false),
    addIntegration: (credentialId: string) =>
      grant({ expertId, data: { credential_ids: [credentialId] } }),
    removeIntegration: (credentialId: string) =>
      revoke({ expertId, credentialId }),
    isGranting,
    isRevoking,
  };
}
