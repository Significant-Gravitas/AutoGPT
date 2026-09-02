"use client";

import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV2ListChatConnectionsQueryKey,
  getGetV2ListChatTransportsQueryKey,
  useGetV2ListChatConnections,
  usePutV2SetDefaultChatTransport,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import { toast } from "@/components/molecules/Toast/use-toast";

import { routeOf, visibleOffers } from "./helpers";

export function useAIConnectionsSection() {
  const queryClient = useQueryClient();

  // The offers endpoint rather than transports: transports answers "what may
  // run", which is enough to route a turn and not enough to render one. What
  // a connection is called, what backs it, and which models each tier uses
  // are product statements, and a client that derives them drifts from the
  // server that enforces them.
  const connectionsQuery = useGetV2ListChatConnections({
    query: { refetchOnWindowFocus: true },
  });

  const offers: AIConnectionOffer[] = visibleOffers(
    connectionsQuery.data?.status === 200
      ? connectionsQuery.data.data.offers
      : undefined,
  );

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

  function accountFor(offer: AIConnectionOffer): string | undefined {
    if (!offer.credential_id) return undefined;
    return accountByCredentialId.get(offer.credential_id);
  }

  const { mutate: setDefault, isPending: isSaving } =
    usePutV2SetDefaultChatTransport({
      mutation: {
        onSuccess: () => {
          // Both lists carry the default: offers renders it, transports
          // routes it. Refreshing one and not the other leaves the screen
          // disagreeing with what a new chat will actually do.
          queryClient.invalidateQueries({
            queryKey: getGetV2ListChatConnectionsQueryKey(),
          });
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
  const selectedKey =
    offers.find((offer) => offer.is_default)?.offer_id ?? null;

  function chooseDefault(offer: AIConnectionOffer) {
    if (offer.is_default) return;
    setDefault({ data: routeOf(offer) });
  }

  return {
    connections: offers,
    accountFor,
    selectedKey,
    chooseDefault,
    isSaving,
    isLoading: connectionsQuery.isLoading,
    isError: connectionsQuery.isError,
    refetch: connectionsQuery.refetch,
  };
}
