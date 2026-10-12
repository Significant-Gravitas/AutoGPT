"use client";

import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV2GetSessionQueryKey,
  getGetV2ListChatConnectionsQueryKey,
  useGetV2ListChatConnections,
  usePutV2ChangeTheConnectionAnExistingChatRunsOn,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";

import type { ProviderFailure } from "../../providerFailure";
import { alternativeConnection, formatResetHint } from "./helpers";

interface Args {
  failure: ProviderFailure | null;
  sessionId: string | null;
  onDismiss: () => void;
}

export function useProviderLimitDialog({
  failure,
  sessionId,
  onDismiss,
}: Args) {
  const queryClient = useQueryClient();

  const connectionsQuery = useGetV2ListChatConnections({
    query: { enabled: Boolean(failure) },
  });
  const offers =
    connectionsQuery.data?.status === 200
      ? connectionsQuery.data.data.offers
      : undefined;

  // "There is no other connection" is a claim about the account, and while the
  // list is still loading -- or failed to load -- we do not know that. Saying
  // it anyway tells a user with a perfectly good second connection that they
  // have none, at the moment they most need it.
  const isLoadingOffers = connectionsQuery.isLoading;
  const failedToLoadOffers =
    connectionsQuery.isError ||
    (connectionsQuery.data !== undefined &&
      connectionsQuery.data.status !== 200);
  const alternative = alternativeConnection(offers, failure);

  const { mutateAsync: changeConnection, isPending: isSwitching } =
    usePutV2ChangeTheConnectionAnExistingChatRunsOn({
      mutation: {
        onSuccess: async () => {
          // The composer already holds the message the provider refused, so
          // the user resends when they are ready. Never automatic: the whole
          // point is that moving the bill is their call.
          await Promise.all([
            queryClient.invalidateQueries({
              queryKey: getGetV2ListChatConnectionsQueryKey(),
            }),
            queryClient.invalidateQueries({
              queryKey: getGetV2GetSessionQueryKey(sessionId ?? undefined),
            }),
          ]);
          onDismiss();
        },
        onError: () => {
          toast({
            variant: "destructive",
            title: "Could not switch connection",
            description:
              "This chat is still on the connection it was. Try again, or pick one in Settings.",
          });
        },
      },
    });

  async function continueHere() {
    if (!sessionId || !alternative) return;
    try {
      await changeConnection({
        sessionId,
        data: {
          llm_auth_provider: alternative.auth_provider,
          llm_credential_id: alternative.credential_id,
        },
      });
    } catch {
      // React Query's onError handler above owns the user-facing failure.
      // Consume mutateAsync's rejection so a button click does not become an
      // unhandled promise rejection as well.
    }
  }

  return {
    alternative,
    isLoadingOffers,
    failedToLoadOffers,
    retryOffers: () => void connectionsQuery.refetch(),
    continueHere,
    isSwitching,
    resetHint: formatResetHint(failure?.resetsAt ?? null),
  };
}
