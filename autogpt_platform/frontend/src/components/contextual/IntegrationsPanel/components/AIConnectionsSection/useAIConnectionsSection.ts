"use client";

import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV2ListChatTransportsQueryKey,
  useGetV2ListChatTransports,
  usePutV2SetDefaultChatTransport,
} from "@/app/api/__generated__/endpoints/chat/chat";
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
    selectedKey,
    chooseDefault,
    isSaving,
    isLoading: transportsQuery.isLoading,
    isError: transportsQuery.isError,
    refetch: transportsQuery.refetch,
  };
}
