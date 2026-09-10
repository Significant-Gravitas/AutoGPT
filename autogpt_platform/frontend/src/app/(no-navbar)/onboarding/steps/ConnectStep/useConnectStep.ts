"use client";

import {
  getGetV2ListChatConnectionsQueryKey,
  useGetV2ListChatConnections,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { useQueryClient } from "@tanstack/react-query";
import { useOAuthConnect } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useOAuthConnect";

import { useOnboardingWizardStore } from "../../store";
import { hasLinkedSubscription } from "./helpers";

export function useConnectStep() {
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const queryClient = useQueryClient();

  const connectionsQuery = useGetV2ListChatConnections({
    query: { refetchOnWindowFocus: false },
  });
  const offers =
    connectionsQuery.data?.status === 200
      ? connectionsQuery.data.data.offers
      : undefined;

  // A successful sign-in does not move the wizard on: the box turns to
  // "Connected" and the user goes on with Next when they are ready.
  const { connect, isPending } = useOAuthConnect({
    provider: "codex",
    onSuccess: () =>
      void queryClient.invalidateQueries({
        queryKey: getGetV2ListChatConnectionsQueryKey(),
      }),
  });

  return {
    connect,
    isConnecting: isPending,
    skip: nextStep,
    isAlreadyLinked: hasLinkedSubscription(offers),
  };
}
