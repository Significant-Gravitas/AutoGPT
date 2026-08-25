"use client";

import { useQueryClient } from "@tanstack/react-query";

import {
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

  const alternative = alternativeConnection(offers, failure);

  const { mutateAsync: changeConnection, isPending: isSwitching } =
    usePutV2ChangeTheConnectionAnExistingChatRunsOn({
      mutation: {
        onSuccess: () => {
          // The composer already holds the message the provider refused, so
          // the user resends when they are ready. Never automatic: the whole
          // point is that moving the bill is their call.
          queryClient.invalidateQueries({
            queryKey: getGetV2ListChatConnectionsQueryKey(),
          });
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
    await changeConnection({
      sessionId,
      data: {
        llm_auth_provider: alternative.auth_provider,
        llm_credential_id: alternative.credential_id,
      },
    });
  }

  return {
    alternative,
    continueHere,
    isSwitching,
    resetHint: formatResetHint(failure?.resetsAt ?? null),
  };
}
