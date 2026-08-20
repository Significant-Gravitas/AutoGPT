"use client";

import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV2ListChatTransportsQueryKey,
  useGetV2ListChatTransports,
  usePutV2SetDefaultChatTransport,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import { toast } from "@/components/molecules/Toast/use-toast";

import { isSelectable, transportKey } from "./helpers";

export function useAIConnectionsSection() {
  const queryClient = useQueryClient();
  const transportsQuery = useGetV2ListChatTransports({
    query: { refetchOnWindowFocus: true },
  });

  const transports: ChatTransportResponse[] =
    transportsQuery.data?.status === 200
      ? transportsQuery.data.data.transports
      : [];
  const connections = transports.filter(isSelectable);

  // The account a connection runs as. Read from the stored credential rather
  // than asked of the provider: this is the identity the user linked, so it is
  // true without a live call, and it cannot go stale the way a usage window
  // can.
  const credentialsQuery = useGetV1ListCredentials({
    query: {
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });
  const accountByCredentialId = new Map(
    (credentialsQuery.data ?? [])
      .filter((credential) => credential.username)
      .map((credential) => [credential.id, credential.username as string]),
  );

  function accountFor(transport: ChatTransportResponse): string | undefined {
    if (!transport.credential_id) return undefined;
    return accountByCredentialId.get(transport.credential_id);
  }

  const { mutateAsync: setDefault, isPending: isSaving } =
    usePutV2SetDefaultChatTransport({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getGetV2ListChatTransportsQueryKey(),
          });
        },
        onError: () => {
          toast({
            variant: "destructive",
            title: "Could not save that connection",
            description:
              "Your default is unchanged. Check the connection is still linked, then try again.",
          });
        },
      },
    });

  // Tracked separately from the query so the row the user clicked shows the
  // pending state, rather than every row going busy at once.
  const selectedKey = connections.find((t) => t.default)
    ? transportKey(connections.find((t) => t.default)!)
    : null;

  async function chooseDefault(transport: ChatTransportResponse) {
    if (transport.default) return;
    await setDefault({
      data: {
        auth_provider: transport.auth_provider,
        credential_id: transport.credential_id,
      },
    });
  }

  return {
    connections,
    accountFor,
    selectedKey,
    chooseDefault,
    isSaving,
    isLoading: transportsQuery.isLoading,
    isError: transportsQuery.isError,
    refetch: transportsQuery.refetch,
  };
}
